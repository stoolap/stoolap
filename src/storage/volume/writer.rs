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

//! Volume writer: freezes in-memory rows into a column-major frozen volume.
//!
//! The freeze operation takes a set of rows (from the hot buffer or snapshot
//! recovery) and converts them to column-major storage with zone maps and
//! pre-computed aggregate stats. This is done by a background thread during
//! the seal operation.

use ahash::AHashMap;

use crate::common::SmartString;
use crate::core::{DataType, Row, Schema, Value};

use super::column::{ColumnData, ZoneMap};
use super::stats::VolumeAggregateStats;

/// A frozen volume ready for queries.
///
/// This is the in-memory representation. Serialization to/from disk
/// will be added in a separate module.
pub struct FrozenVolume {
    /// Column data stored as typed arrays
    pub columns: Vec<ColumnData>,
    /// Zone maps per column
    pub zone_maps: Vec<ZoneMap>,
    /// Bloom filters per column (for fast equality membership testing)
    pub bloom_filters: Vec<super::column::ColumnBloomFilter>,
    /// Pre-computed aggregate stats
    pub stats: VolumeAggregateStats,
    /// Number of live rows
    pub row_count: usize,
    /// Column names (from schema)
    pub column_names: Vec<String>,
    /// Column types (from schema)
    pub column_types: Vec<DataType>,
    /// Row IDs for each row (preserves original IDs for index compatibility)
    pub row_ids: Vec<i64>,
    /// Whether the time/integer columns are sorted (enables binary search)
    pub sorted_columns: Vec<bool>,
    /// Precomputed lowercase column name → index map for O(1) lookup.
    /// Built once at construction; replaces O(C) linear scan in column_index().
    pub column_name_map: AHashMap<SmartString, usize>,
    /// Per-volume unique hash index: lazily built, never invalidated (volume is immutable).
    /// Key: sorted column indices for a UNIQUE constraint.
    /// Value: hash(column values) → row indices within this volume.
    /// Multiple indices per hash when the volume has duplicate values (pre-existing
    /// dupes that haven't been cleaned yet). Typically 1 entry per hash.
    #[allow(clippy::type_complexity)]
    pub unique_indices: parking_lot::RwLock<
        rustc_hash::FxHashMap<Vec<usize>, rustc_hash::FxHashMap<u64, Vec<u32>>>,
    >,
}

/// Builder that accumulates rows and produces a FrozenVolume.
pub struct VolumeBuilder {
    schema: Schema,
    num_cols: usize,
    // Per-column accumulators
    int_cols: Vec<Vec<i64>>,
    float_cols: Vec<Vec<f64>>,
    ts_cols: Vec<Vec<i64>>, // nanos since epoch
    bool_cols: Vec<Vec<bool>>,
    dict_cols: Vec<Vec<u32>>,
    #[allow(clippy::type_complexity)]
    bytes_cols: Vec<(Vec<u8>, Vec<(u64, u64)>)>, // (data, offsets)
    null_cols: Vec<Vec<bool>>,
    // Column type mapping
    col_storage: Vec<StorageKind>,
    // Dictionary maps for text columns
    dict_maps: Vec<AHashMap<SmartString, u32>>,
    dict_tables: Vec<Vec<SmartString>>,
    // Zone maps
    zone_maps: Vec<ZoneMap>,
    // Stats
    stats: VolumeAggregateStats,
    // Row IDs
    row_ids: Vec<i64>,
    // Sort tracking
    last_values: Vec<Option<i64>>,
    sorted: Vec<bool>,
    // Row count
    row_count: usize,
}

#[derive(Clone, Copy)]
enum StorageKind {
    Int64(usize),           // index into int_cols
    Float64(usize),         // index into float_cols
    Timestamp(usize),       // index into ts_cols
    Boolean(usize),         // index into bool_cols
    Dictionary(usize),      // index into dict_cols
    Bytes(usize, DataType), // index into bytes_cols + ext type
}

impl VolumeBuilder {
    /// Create a new builder from a table schema.
    pub fn new(schema: &Schema) -> Self {
        let num_cols = schema.columns.len();
        let mut int_cols = Vec::new();
        let mut float_cols = Vec::new();
        let mut ts_cols = Vec::new();
        let mut bool_cols = Vec::new();
        let mut dict_cols = Vec::new();
        let mut bytes_cols = Vec::new();
        let mut col_storage = Vec::with_capacity(num_cols);
        let mut last_values = Vec::with_capacity(num_cols);
        let mut sorted = Vec::with_capacity(num_cols);

        for col in &schema.columns {
            match col.data_type {
                DataType::Integer => {
                    let idx = int_cols.len();
                    int_cols.push(Vec::new());
                    col_storage.push(StorageKind::Int64(idx));
                    last_values.push(None);
                    sorted.push(true);
                }
                DataType::Float => {
                    let idx = float_cols.len();
                    float_cols.push(Vec::new());
                    col_storage.push(StorageKind::Float64(idx));
                    last_values.push(None);
                    sorted.push(false); // floats: don't track sort
                }
                DataType::Timestamp => {
                    let idx = ts_cols.len();
                    ts_cols.push(Vec::new());
                    col_storage.push(StorageKind::Timestamp(idx));
                    last_values.push(None);
                    sorted.push(true);
                }
                DataType::Boolean => {
                    let idx = bool_cols.len();
                    bool_cols.push(Vec::new());
                    col_storage.push(StorageKind::Boolean(idx));
                    last_values.push(None);
                    sorted.push(false);
                }
                DataType::Text => {
                    let idx = dict_cols.len();
                    dict_cols.push(Vec::new());
                    col_storage.push(StorageKind::Dictionary(idx));
                    last_values.push(None);
                    sorted.push(false);
                }
                dt => {
                    // JSON, Vector, etc. → raw bytes
                    let idx = bytes_cols.len();
                    bytes_cols.push((Vec::new(), Vec::new()));
                    col_storage.push(StorageKind::Bytes(idx, dt));
                    last_values.push(None);
                    sorted.push(false);
                }
            }
        }

        let num_dict_cols = dict_cols.len();
        Self {
            schema: schema.clone(),
            num_cols,
            int_cols,
            float_cols,
            ts_cols,
            bool_cols,
            dict_cols,
            bytes_cols,
            null_cols: vec![Vec::new(); num_cols],
            col_storage,
            dict_maps: vec![AHashMap::new(); num_dict_cols],
            dict_tables: vec![Vec::new(); num_dict_cols],
            zone_maps: (0..num_cols)
                .map(|_| ZoneMap {
                    min: Value::Null(DataType::Null),
                    max: Value::Null(DataType::Null),
                    null_count: 0,
                    row_count: 0,
                })
                .collect(),
            stats: VolumeAggregateStats::new(num_cols),
            row_ids: Vec::new(),
            last_values,
            sorted,
            row_count: 0,
        }
    }

    /// Create a builder with pre-allocated capacity.
    pub fn with_capacity(schema: &Schema, capacity: usize) -> Self {
        let mut builder = Self::new(schema);
        builder.row_ids.reserve(capacity);
        for nulls in &mut builder.null_cols {
            nulls.reserve(capacity);
        }
        for v in &mut builder.int_cols {
            v.reserve(capacity);
        }
        for v in &mut builder.float_cols {
            v.reserve(capacity);
        }
        for v in &mut builder.ts_cols {
            v.reserve(capacity);
        }
        for v in &mut builder.bool_cols {
            v.reserve(capacity);
        }
        for v in &mut builder.dict_cols {
            v.reserve(capacity);
        }
        builder
    }

    /// Add a row to the volume.
    pub fn add_row(&mut self, row_id: i64, row: &Row) {
        self.row_ids.push(row_id);
        self.stats.total_rows += 1;
        self.stats.live_rows += 1;

        for col_idx in 0..self.num_cols {
            let value = row.get(col_idx).unwrap_or(&Value::Null(DataType::Null));

            self.zone_maps[col_idx].row_count += 1;
            let is_null = value.is_null();
            self.null_cols[col_idx].push(is_null);

            if is_null {
                self.zone_maps[col_idx].null_count += 1;
                // NULL placeholder (0) breaks sorted-order invariant that
                // binary search requires. Mark column unsorted.
                self.sorted[col_idx] = false;
                // Push placeholder for null
                match self.col_storage[col_idx] {
                    StorageKind::Int64(idx) => self.int_cols[idx].push(0),
                    StorageKind::Float64(idx) => self.float_cols[idx].push(0.0),
                    StorageKind::Timestamp(idx) => self.ts_cols[idx].push(0),
                    StorageKind::Boolean(idx) => self.bool_cols[idx].push(false),
                    StorageKind::Dictionary(idx) => self.dict_cols[idx].push(0),
                    StorageKind::Bytes(idx, _) => {
                        self.bytes_cols[idx].1.push((0, 0));
                    }
                }
                continue;
            }

            // Accumulate stats
            self.stats.columns[col_idx].accumulate(value);

            // Update zone map
            let zm = &mut self.zone_maps[col_idx];
            if zm.min.is_null() {
                zm.min = value.clone();
                zm.max = value.clone();
            } else {
                if let Ok(std::cmp::Ordering::Less) = value.compare(&zm.min) {
                    zm.min = value.clone();
                }
                if let Ok(std::cmp::Ordering::Greater) = value.compare(&zm.max) {
                    zm.max = value.clone();
                }
            }

            // Store in typed column
            match self.col_storage[col_idx] {
                StorageKind::Int64(idx) => {
                    let v = match value {
                        Value::Integer(i) => *i,
                        _ => 0,
                    };
                    // Track sortedness
                    if self.sorted[col_idx] {
                        if let Some(last) = self.last_values[col_idx] {
                            if v < last {
                                self.sorted[col_idx] = false;
                            }
                        }
                        self.last_values[col_idx] = Some(v);
                    }
                    self.int_cols[idx].push(v);
                }
                StorageKind::Float64(idx) => {
                    let v = match value {
                        Value::Float(f) => *f,
                        _ => 0.0,
                    };
                    self.float_cols[idx].push(v);
                }
                StorageKind::Timestamp(idx) => {
                    let nanos = match value {
                        Value::Timestamp(ts) => ts.timestamp_nanos_opt().unwrap_or_else(|| {
                            ts.timestamp()
                                .wrapping_mul(1_000_000_000)
                                .wrapping_add(ts.timestamp_subsec_nanos() as i64)
                        }),
                        _ => 0,
                    };
                    if self.sorted[col_idx] {
                        if let Some(last) = self.last_values[col_idx] {
                            if nanos < last {
                                self.sorted[col_idx] = false;
                            }
                        }
                        self.last_values[col_idx] = Some(nanos);
                    }
                    self.ts_cols[idx].push(nanos);
                }
                StorageKind::Boolean(idx) => {
                    let v = match value {
                        Value::Boolean(b) => *b,
                        _ => false,
                    };
                    self.bool_cols[idx].push(v);
                }
                StorageKind::Dictionary(idx) => {
                    let s = match value {
                        Value::Text(s) => s.clone(),
                        _ => SmartString::from(""),
                    };
                    let dict_id = if let Some(&id) = self.dict_maps[idx].get(&s) {
                        id
                    } else {
                        let id = self.dict_tables[idx].len() as u32;
                        self.dict_tables[idx].push(s.clone());
                        self.dict_maps[idx].insert(s, id);
                        id
                    };
                    self.dict_cols[idx].push(dict_id);
                }
                StorageKind::Bytes(idx, _) => {
                    let bytes = match value {
                        Value::Extension(data) => {
                            if data.len() > 1 {
                                &data[1..] // skip type tag
                            } else {
                                &[]
                            }
                        }
                        _ => &[],
                    };
                    let offset = self.bytes_cols[idx].0.len() as u64;
                    let length = bytes.len() as u64;
                    self.bytes_cols[idx].0.extend_from_slice(bytes);
                    self.bytes_cols[idx].1.push((offset, length));
                }
            }
        }
        self.row_count += 1;
    }

    /// Freeze the builder into a FrozenVolume.
    pub fn finish(mut self) -> FrozenVolume {
        let mut columns = Vec::with_capacity(self.num_cols);
        let mut sorted_columns = Vec::with_capacity(self.num_cols);

        for col_idx in 0..self.num_cols {
            let nulls = std::mem::take(&mut self.null_cols[col_idx]);
            sorted_columns.push(self.sorted[col_idx]);

            let col_data = match self.col_storage[col_idx] {
                StorageKind::Int64(idx) => ColumnData::Int64 {
                    values: std::mem::take(&mut self.int_cols[idx]),
                    nulls,
                },
                StorageKind::Float64(idx) => ColumnData::Float64 {
                    values: std::mem::take(&mut self.float_cols[idx]),
                    nulls,
                },
                StorageKind::Timestamp(idx) => ColumnData::TimestampNanos {
                    values: std::mem::take(&mut self.ts_cols[idx]),
                    nulls,
                },
                StorageKind::Boolean(idx) => ColumnData::Boolean {
                    values: std::mem::take(&mut self.bool_cols[idx]),
                    nulls,
                },
                StorageKind::Dictionary(idx) => ColumnData::Dictionary {
                    ids: std::mem::take(&mut self.dict_cols[idx]),
                    dictionary: std::mem::take(&mut self.dict_tables[idx]),
                    nulls,
                },
                StorageKind::Bytes(idx, ext_type) => {
                    let (data, offsets) = std::mem::take(&mut self.bytes_cols[idx]);
                    ColumnData::Bytes {
                        data,
                        offsets,
                        ext_type,
                        nulls,
                    }
                }
            };
            columns.push(col_data);
        }

        let column_names: Vec<String> =
            self.schema.columns.iter().map(|c| c.name.clone()).collect();
        let column_types: Vec<DataType> = self.schema.columns.iter().map(|c| c.data_type).collect();

        // Build bloom filters from column data using typed methods
        // to avoid allocating a Value per cell (saves ~500K allocs for 100K rows).
        let bloom_filters: Vec<super::column::ColumnBloomFilter> = columns
            .iter()
            .map(|col| {
                let mut bf = super::column::ColumnBloomFilter::new(self.row_count.max(1));
                for i in 0..self.row_count {
                    if col.is_null(i) {
                        continue;
                    }
                    match col {
                        super::column::ColumnData::Int64 { values, .. } => {
                            bf.add_i64(values[i]);
                        }
                        super::column::ColumnData::Float64 { values, .. } => {
                            bf.add_f64(values[i]);
                        }
                        super::column::ColumnData::TimestampNanos { values, .. } => {
                            bf.add_timestamp_nanos(values[i]);
                        }
                        super::column::ColumnData::Boolean { values, .. } => {
                            bf.add_bool(values[i]);
                        }
                        super::column::ColumnData::Dictionary {
                            ids, dictionary, ..
                        } => {
                            let dict_id = ids[i] as usize;
                            if dict_id < dictionary.len() {
                                bf.add_str(dictionary[dict_id].as_str());
                            }
                        }
                        super::column::ColumnData::Bytes { .. } => {
                            // Fall back to Value for complex types
                            let value = col.get_value(i);
                            bf.add(&value);
                        }
                    }
                }
                bf
            })
            .collect();

        // Ensure row_ids are sorted — binary_search in manifest/table depends on this.
        // All production paths (seal via BTree iter, compact via explicit sort, snapshot
        // recovery via BTreeMap iter) provide rows in ascending row_id order. This
        // check catches any future caller that violates this invariant.
        // Using a regular check (not debug_assert) because silent corruption in
        // release builds from unsorted row_ids would be catastrophic.
        if !self.row_ids.windows(2).all(|w| w[0] < w[1]) {
            // Sort as fallback instead of panicking
            self.row_ids.sort_unstable();
        }

        let column_name_map: AHashMap<SmartString, usize> = column_names
            .iter()
            .enumerate()
            .map(|(i, name)| (SmartString::from(name.to_lowercase()), i))
            .collect();

        FrozenVolume {
            columns,
            zone_maps: self.zone_maps,
            bloom_filters,
            stats: self.stats,
            row_count: self.row_count,
            column_names,
            column_types,
            row_ids: self.row_ids,
            sorted_columns,
            column_name_map,
            unique_indices: parking_lot::RwLock::new(rustc_hash::FxHashMap::default()),
        }
    }
}

impl FrozenVolume {
    /// Get a row as a Vec of Values (for executor compatibility).
    pub fn get_row(&self, idx: usize) -> Row {
        let values: Vec<Value> = self.columns.iter().map(|col| col.get_value(idx)).collect();
        Row::from_values(values)
    }

    /// Get specific columns of a row (projection pushdown).
    pub fn get_row_projected(&self, idx: usize, col_indices: &[usize]) -> Row {
        let values: Vec<Value> = col_indices
            .iter()
            .map(|&col| self.columns[col].get_value(idx))
            .collect();
        Row::from_values(values)
    }

    /// Check if a column is sorted (enables binary search).
    #[inline]
    pub fn is_sorted(&self, col_idx: usize) -> bool {
        self.sorted_columns[col_idx]
    }

    /// Look up a composite unique key in this volume's per-volume hash index.
    /// Calls `f` for each matching row index. Supports volumes with duplicate values
    /// (pre-existing dupes not yet cleaned). The caller decides which match to accept
    /// (e.g., skip tombstoned rows, take first non-tombstoned).
    ///
    /// The index is built lazily on first call per column set and never invalidated
    /// (volume is immutable). Build cost: O(K) where K = this volume's row_count.
    /// Lookup cost: O(1) amortized.
    pub fn unique_lookup_all(
        &self,
        col_indices: &[usize],
        values: &[&Value],
        mut f: impl FnMut(u32) -> bool, // return true to stop early
    ) {
        use std::hash::{Hash, Hasher};

        if col_indices.iter().any(|&idx| idx >= self.columns.len()) {
            return;
        }

        // Compute hash of query values
        let mut hasher = ahash::AHasher::default();
        for &val in values {
            val.hash(&mut hasher);
        }
        let hash = hasher.finish();

        // Fast path: check if index is already built
        {
            let indices = self.unique_indices.read();
            if let Some(idx_map) = indices.get(col_indices) {
                if let Some(row_indices) = idx_map.get(&hash) {
                    for &row_idx in row_indices {
                        // Verify actual values match (handle hash collision)
                        let matches = col_indices.iter().zip(values.iter()).all(|(&ci, &val)| {
                            let vol_val = self.columns[ci].get_value(row_idx as usize);
                            !vol_val.is_null() && vol_val == *val
                        });
                        if matches && f(row_idx) {
                            return;
                        }
                    }
                }
                return;
            }
        }

        // Build index for this column set (first use)
        let mut idx_map: rustc_hash::FxHashMap<u64, Vec<u32>> =
            rustc_hash::FxHashMap::with_capacity_and_hasher(self.row_count, Default::default());
        for row_idx in 0..self.row_count {
            let mut row_hasher = ahash::AHasher::default();
            let mut has_null = false;
            for &ci in col_indices {
                if self.columns[ci].is_null(row_idx) {
                    has_null = true;
                    break;
                }
                self.columns[ci].get_value(row_idx).hash(&mut row_hasher);
            }
            if has_null {
                continue;
            }
            let row_hash = row_hasher.finish();
            idx_map.entry(row_hash).or_default().push(row_idx as u32);
        }

        // Look up before storing
        if let Some(row_indices) = idx_map.get(&hash) {
            for &row_idx in row_indices {
                let matches = col_indices.iter().zip(values.iter()).all(|(&ci, &val)| {
                    let vol_val = self.columns[ci].get_value(row_idx as usize);
                    !vol_val.is_null() && vol_val == *val
                });
                if matches && f(row_idx) {
                    break;
                }
            }
        }

        // Store the built index
        self.unique_indices
            .write()
            .insert(col_indices.to_vec(), idx_map);
    }

    /// Find the column index by name. O(1) via precomputed hashmap.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        // Fast path: use the precomputed map (handles lowercase input directly)
        if let Some(&idx) = self.column_name_map.get(name) {
            return Some(idx);
        }
        // Fallback for mixed-case input: lowercase then lookup
        let lower = name.to_lowercase();
        self.column_name_map.get(lower.as_str()).copied()
    }

    /// Get a row normalized to a (possibly different) schema.
    ///
    /// Handles schema evolution:
    /// - Matching columns (by name): returned as-is
    /// - Renamed columns: matched by position + type when name lookup fails
    /// - New columns (not in volume): filled with DEFAULT or NULL
    /// - Dropped columns (not in current schema): skipped
    pub fn get_row_normalized(&self, idx: usize, current_schema: &Schema) -> Row {
        let mut values = Vec::with_capacity(current_schema.columns.len());
        for (pos, col) in current_schema.columns.iter().enumerate() {
            // First try name-based matching
            if let Some(vol_idx) = self.column_index(&col.name_lower) {
                values.push(self.columns[vol_idx].get_value(idx));
            } else if pos < self.columns.len()
                && pos < self.column_types.len()
                && self.column_types[pos] == col.data_type
            {
                // Positional fallback: same position + same type = likely a rename
                values.push(self.columns[pos].get_value(idx));
            } else {
                // Column added after this volume was created
                if let Some(ref default_val) = col.default_value {
                    values.push(default_val.clone());
                } else {
                    values.push(Value::Null(col.data_type));
                }
            }
        }
        Row::from_values(values)
    }

    /// Estimate the in-memory size of this volume in bytes.
    pub fn memory_size(&self) -> usize {
        let mut size = 0;
        for col in &self.columns {
            size += match col {
                ColumnData::Int64 { values, nulls } => values.len() * 8 + nulls.len(),
                ColumnData::Float64 { values, nulls } => values.len() * 8 + nulls.len(),
                ColumnData::TimestampNanos { values, nulls } => values.len() * 8 + nulls.len(),
                ColumnData::Boolean { values, nulls } => values.len() + nulls.len(),
                ColumnData::Dictionary {
                    ids,
                    dictionary,
                    nulls,
                } => {
                    ids.len() * 4
                        + dictionary.iter().map(|s| s.len() + 24).sum::<usize>()
                        + nulls.len()
                }
                ColumnData::Bytes {
                    data,
                    offsets,
                    nulls,
                    ..
                } => data.len() + offsets.len() * 16 + nulls.len(),
            };
        }
        size + self.row_ids.len() * 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SchemaBuilder;

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("time", DataType::Timestamp, false, false)
            .column("exchange", DataType::Text, false, false)
            .column("price", DataType::Float, false, false)
            .build()
    }

    #[test]
    fn test_freeze_basic() {
        let schema = test_schema();
        let mut builder = VolumeBuilder::with_capacity(&schema, 3);

        let ts1 = chrono::Utc::now();
        let ts2 = ts1 + chrono::Duration::minutes(1);
        let ts3 = ts2 + chrono::Duration::minutes(1);

        builder.add_row(
            1,
            &Row::from_values(vec![
                Value::Integer(1),
                Value::Timestamp(ts1),
                Value::text("binance"),
                Value::Float(100.0),
            ]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![
                Value::Integer(2),
                Value::Timestamp(ts2),
                Value::text("coinbase"),
                Value::Float(101.5),
            ]),
        );
        builder.add_row(
            3,
            &Row::from_values(vec![
                Value::Integer(3),
                Value::Timestamp(ts3),
                Value::text("binance"),
                Value::Float(99.0),
            ]),
        );

        let volume = builder.finish();

        assert_eq!(volume.row_count, 3);
        assert_eq!(volume.columns.len(), 4);
        assert_eq!(volume.stats.count_star(), 3);

        // Check typed access
        assert_eq!(volume.columns[0].get_i64(0), 1);
        assert_eq!(volume.columns[0].get_i64(2), 3);
        assert_eq!(volume.columns[3].get_f64(1), 101.5);

        // Check dictionary encoding
        assert_eq!(volume.columns[2].get_str(0), "binance");
        assert_eq!(volume.columns[2].get_str(1), "coinbase");
        assert_eq!(volume.columns[2].get_str(2), "binance");
        // binance appears twice but uses same dict ID
        assert_eq!(
            volume.columns[2].get_dict_id(0),
            volume.columns[2].get_dict_id(2)
        );

        // Check zone maps
        assert_eq!(volume.zone_maps[0].min, Value::Integer(1));
        assert_eq!(volume.zone_maps[0].max, Value::Integer(3));
        assert_eq!(volume.zone_maps[3].min, Value::Float(99.0));
        assert_eq!(volume.zone_maps[3].max, Value::Float(101.5));

        // Check stats
        assert_eq!(volume.stats.sum(3), 300.5); // 100.0 + 101.5 + 99.0

        // Check sortedness
        assert!(volume.is_sorted(0)); // id is sorted
        assert!(volume.is_sorted(1)); // time is sorted

        // Check row reconstruction
        let row = volume.get_row(0);
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(2), Some(&Value::text("binance")));
    }

    #[test]
    fn test_freeze_with_nulls() {
        let schema = test_schema();
        let mut builder = VolumeBuilder::new(&schema);

        builder.add_row(
            1,
            &Row::from_values(vec![
                Value::Integer(1),
                Value::Null(DataType::Timestamp),
                Value::text("binance"),
                Value::Null(DataType::Float),
            ]),
        );

        let volume = builder.finish();
        assert!(volume.columns[1].is_null(0));
        assert!(volume.columns[3].is_null(0));
        assert!(!volume.columns[0].is_null(0));

        let row = volume.get_row(0);
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert!(row.get(1).unwrap().is_null());
    }

    #[test]
    fn test_binary_search_on_sorted() {
        let schema = SchemaBuilder::new("test")
            .column("time", DataType::Timestamp, false, false)
            .build();
        let mut builder = VolumeBuilder::new(&schema);

        let base = chrono::Utc::now();
        for i in 0..100 {
            let ts = base + chrono::Duration::minutes(i);
            builder.add_row(i, &Row::from_values(vec![Value::Timestamp(ts)]));
        }

        let volume = builder.finish();
        assert!(volume.is_sorted(0));

        // Binary search for row 50
        let target_nanos = {
            let ts = base + chrono::Duration::minutes(50);
            ts.timestamp_nanos_opt()
                .unwrap_or(ts.timestamp() * 1_000_000_000)
        };
        let idx = volume.columns[0].binary_search_ge(target_nanos);
        assert_eq!(idx, 50);
    }

    #[test]
    fn test_projection() {
        let schema = test_schema();
        let mut builder = VolumeBuilder::new(&schema);

        builder.add_row(
            1,
            &Row::from_values(vec![
                Value::Integer(1),
                Value::Timestamp(chrono::Utc::now()),
                Value::text("binance"),
                Value::Float(100.0),
            ]),
        );

        let volume = builder.finish();

        // Project only id and price (columns 0 and 3)
        let row = volume.get_row_projected(0, &[0, 3]);
        assert_eq!(row.len(), 2);
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(1), Some(&Value::Float(100.0)));
    }
}
