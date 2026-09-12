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

use std::sync::Arc;
use std::sync::OnceLock;

use ahash::AHashMap;

/// Global eviction epoch. Updated by MVCCEngine::evict_idle_volumes(),
/// read by VolumeScanner to stamp last_access_epoch correctly.
/// Using a global avoids threading the epoch through every scan path.
pub static GLOBAL_EVICTION_EPOCH: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

use crate::common::SmartString;
use crate::core::{DataType, Row, Schema, Value};
use crate::storage::mvcc::read_memory::{DeferredExports, PayloadCharge};

use super::column::{ColumnData, ZoneMap, ROW_GROUP_SIZE};
use super::format::{
    deserialize_column_block, deserialize_column_block_into, serialize_column_block, COL_BYTES,
    COL_DICTIONARY,
};
use super::stats::VolumeAggregateStats;

// =============================================================================
// CompressedBlockStore: per-column per-row-group LZ4 blocks in RAM
// =============================================================================

/// Holds LZ4-compressed column data in RAM. Each column is split into row-group-
/// sized blocks (64K rows). Decompression from RAM runs at ~4 GB/s, negligible
/// compared to disk I/O. This is the backing store for LazyColumns.
pub struct CompressedBlockStore {
    /// blocks[col_idx][group_idx] = LZ4-compressed bytes (no size prefix)
    blocks: Vec<Vec<Vec<u8>>>,
    /// decompressed_lens[col_idx][group_idx] = exact decompressed size
    decompressed_lens: Vec<Vec<usize>>,
    /// Column type tags (COL_INT64, COL_FLOAT64, etc.) for deserialization
    col_type_tags: Vec<u8>,
    /// Column data types
    #[allow(dead_code)]
    col_data_types: Vec<DataType>,
    /// Ext type per column (only meaningful for COL_BYTES columns)
    col_ext_types: Vec<u8>,
    /// Per dictionary column: pre-built Arc, shared across all group decompressions.
    /// Created once at construction — group decompression clones the Arc (~5ns)
    /// instead of cloning all dictionary strings per group.
    col_dicts: Vec<(usize, Arc<[SmartString]>)>,
    /// Row group size (ROW_GROUP_SIZE = 65536)
    group_size: usize,
    /// Total row count across all groups
    row_count: usize,
    /// Process-unique id, the key of this store's entries in the decoded
    /// group cache
    id: usize,
}

/// Ids for block stores; the decoded group cache keys its entries by them
fn next_store_id() -> usize {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl Drop for CompressedBlockStore {
    fn drop(&mut self) {
        super::group_cache::DECODED_GROUPS.remove_store(self.id);
    }
}

impl CompressedBlockStore {
    /// One column of one row group, decoded once and shared through the
    /// decoded group cache while the budget holds it
    pub fn group_column(
        &self,
        col_idx: usize,
        group_idx: usize,
    ) -> std::io::Result<Arc<ColumnData>> {
        let key_col = u32::try_from(col_idx).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "column index out of range",
            )
        })?;
        let key_group = u32::try_from(group_idx).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "group index out of range")
        })?;
        super::group_cache::DECODED_GROUPS.get_or_decode((self.id, key_col, key_group), || {
            self.decompress_single_group(col_idx, group_idx)
        })
    }

    /// Compress existing columns into per-group LZ4 blocks.
    /// Used when sealing (VolumeBuilder::finish() → eager columns → V4 write).
    pub fn compress_columns(
        columns: &LazyColumns,
        col_data_types: &[DataType],
        row_count: usize,
    ) -> std::io::Result<Self> {
        Self::compress_columns_opts(columns, col_data_types, row_count, true)
    }

    /// Build per-group blocks from columns. When `compress` is true, blocks are
    /// LZ4-compressed (blocks that don't shrink are stored raw). When false,
    /// all blocks are stored raw (same V4 layout, no LZ4 overhead).
    pub fn compress_columns_opts(
        columns: &LazyColumns,
        col_data_types: &[DataType],
        row_count: usize,
        compress: bool,
    ) -> std::io::Result<Self> {
        let group_size = ROW_GROUP_SIZE;
        let col_count = columns.len();
        let num_groups = if row_count == 0 {
            0
        } else {
            row_count.div_ceil(group_size)
        };

        // Phase 1: Sequential — extract per-column metadata (type tags, dict, ext types).
        // Must be sequential because shared_dict accumulates across columns.
        let mut shared_dict: Vec<SmartString> = Vec::new();
        let mut dict_ranges = Vec::new();
        let mut col_type_tags = Vec::with_capacity(col_count);
        let mut col_ext_types = Vec::with_capacity(col_count);

        for col_idx in 0..col_count {
            let col = columns.get(col_idx)?;
            let type_tag = match col {
                ColumnData::Int64 { .. } => super::format::COL_INT64,
                ColumnData::Float64 { .. } => super::format::COL_FLOAT64,
                ColumnData::TimestampNanos { .. } => super::format::COL_TIMESTAMP,
                ColumnData::Boolean { .. } => super::format::COL_BOOLEAN,
                ColumnData::Dictionary { .. } => COL_DICTIONARY,
                ColumnData::Bytes { .. } => COL_BYTES,
            };
            col_type_tags.push(type_tag);
            col_ext_types.push(match col {
                ColumnData::Bytes { ext_type, .. } => *ext_type as u8,
                _ => 0,
            });
            if let ColumnData::Dictionary { dictionary, .. } = col {
                let start = shared_dict.len();
                shared_dict.extend(dictionary.iter().cloned());
                dict_ranges.push((col_idx, start, shared_dict.len()));
            }
        }

        // Phase 2: Serialize column blocks (optionally LZ4-compressed).
        let compress_blocks = |col: &ColumnData| -> (Vec<Vec<u8>>, Vec<usize>) {
            let mut col_blocks = Vec::with_capacity(num_groups);
            let mut col_decomp_lens = Vec::with_capacity(num_groups);
            let mut start = 0;
            while start < row_count {
                let end = (start + group_size).min(row_count);
                let raw = serialize_column_block(col, start, end);
                col_decomp_lens.push(raw.len());
                if compress {
                    let compressed = lz4_flex::compress(&raw);
                    if compressed.len() < raw.len() {
                        col_blocks.push(compressed);
                    } else {
                        col_blocks.push(raw);
                    }
                } else {
                    col_blocks.push(raw);
                }
                start = end;
            }
            (col_blocks, col_decomp_lens)
        };

        #[cfg(feature = "parallel")]
        let (all_blocks, all_decomp_lens) = {
            use rayon::prelude::*;
            let results: Vec<(Vec<Vec<u8>>, Vec<usize>)> = (0..col_count)
                .into_par_iter()
                .map(|col_idx| columns.get(col_idx).map(compress_blocks))
                .collect::<std::io::Result<_>>()?;
            let mut all_blocks = Vec::with_capacity(col_count);
            let mut all_decomp_lens = Vec::with_capacity(col_count);
            for (blocks, lens) in results {
                all_blocks.push(blocks);
                all_decomp_lens.push(lens);
            }
            (all_blocks, all_decomp_lens)
        };

        #[cfg(not(feature = "parallel"))]
        let (all_blocks, all_decomp_lens) = {
            let mut all_blocks = Vec::with_capacity(col_count);
            let mut all_decomp_lens = Vec::with_capacity(col_count);
            for col_idx in 0..col_count {
                let (blocks, lens) = compress_blocks(columns.get(col_idx)?);
                all_blocks.push(blocks);
                all_decomp_lens.push(lens);
            }
            (all_blocks, all_decomp_lens)
        };

        let col_dicts: Vec<(usize, Arc<[SmartString]>)> = dict_ranges
            .iter()
            .map(|(ci, start, end)| (*ci, Arc::from(&shared_dict[*start..*end])))
            .collect();
        Ok(Self {
            blocks: all_blocks,
            decompressed_lens: all_decomp_lens,
            col_type_tags,
            col_data_types: col_data_types.to_vec(),
            col_ext_types,
            col_dicts,
            group_size,
            row_count,
            id: next_store_id(),
        })
    }

    /// Build a CompressedBlockStore from pre-compressed blocks (V4 file read).
    /// No decompression happens — blocks are stored as-is from the file.
    #[allow(clippy::too_many_arguments)]
    pub fn from_raw_blocks(
        blocks: Vec<Vec<Vec<u8>>>,
        decompressed_lens: Vec<Vec<usize>>,
        col_type_tags: Vec<u8>,
        col_data_types: Vec<DataType>,
        col_ext_types: Vec<u8>,
        shared_dict: Vec<SmartString>,
        dict_ranges: Vec<(usize, usize, usize)>,
        group_size: usize,
        row_count: usize,
    ) -> Self {
        let col_dicts: Vec<(usize, Arc<[SmartString]>)> = dict_ranges
            .iter()
            .map(|(ci, start, end)| (*ci, Arc::from(&shared_dict[*start..*end])))
            .collect();
        // shared_dict and dict_ranges are consumed — only col_dicts kept
        Self {
            blocks,
            decompressed_lens,
            col_type_tags,
            col_data_types,
            col_ext_types,
            col_dicts,
            group_size,
            row_count,
            id: next_store_id(),
        }
    }

    /// Decompress a single column from RAM. Concatenates all row-group blocks.
    /// Runs at ~4 GB/s (LZ4 from RAM), typically <1ms per column.
    pub fn decompress_column(&self, col_idx: usize) -> std::io::Result<ColumnData> {
        let col_blocks = self.blocks.get(col_idx).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "column index out of range",
            )
        })?;
        let type_tag = *self.col_type_tags.get(col_idx).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing column type tag")
        })?;
        let ext_type = self
            .col_ext_types
            .get(col_idx)
            .and_then(|&tag| DataType::from_u8(tag))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid extension type tag",
                )
            })?;
        let num_groups = col_blocks.len();
        if self.group_size == 0
            || num_groups != self.row_count.div_ceil(self.group_size)
            || self.decompressed_lens.get(col_idx).map(Vec::len) != Some(num_groups)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid column block geometry",
            ));
        }

        // Find pre-built dictionary Arc for this column (Arc clone = ~5ns)
        let dict: Option<Arc<[SmartString]>> = if type_tag == COL_DICTIONARY {
            self.col_dicts
                .iter()
                .find(|(ci, _)| *ci == col_idx)
                .map(|(_, arc)| Arc::clone(arc))
        } else {
            None
        };

        if num_groups == 1 {
            return self.decompress_block(
                col_idx,
                0,
                &col_blocks[0],
                type_tag,
                num_groups,
                dict,
                ext_type,
            );
        }

        // Validate all group lengths before reserving the full-column buffers.
        let mut max_decomp = 0;
        for gi in 0..num_groups {
            let (len, _) = self.block_layout(col_idx, gi, type_tag, num_groups)?;
            max_decomp = max_decomp.max(len);
        }
        let mut lz4_buf = Vec::new();
        lz4_buf
            .try_reserve_exact(max_decomp)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, e))?;
        Ok(match type_tag {
            super::format::COL_INT64 => {
                let mut all_values = Vec::with_capacity(self.row_count);
                let mut all_nulls = Vec::with_capacity(self.row_count);
                for (gi, block) in col_blocks.iter().enumerate() {
                    self.decompress_block_into(
                        col_idx,
                        gi,
                        block,
                        type_tag,
                        num_groups,
                        &mut lz4_buf,
                        &mut all_nulls,
                        Some(&mut all_values),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )?;
                }
                ColumnData::Int64 {
                    values: all_values,
                    nulls: all_nulls,
                }
            }
            super::format::COL_FLOAT64 => {
                let mut all_values = Vec::with_capacity(self.row_count);
                let mut all_nulls = Vec::with_capacity(self.row_count);
                for (gi, block) in col_blocks.iter().enumerate() {
                    self.decompress_block_into(
                        col_idx,
                        gi,
                        block,
                        type_tag,
                        num_groups,
                        &mut lz4_buf,
                        &mut all_nulls,
                        None,
                        Some(&mut all_values),
                        None,
                        None,
                        None,
                        None,
                    )?;
                }
                ColumnData::Float64 {
                    values: all_values,
                    nulls: all_nulls,
                }
            }
            super::format::COL_TIMESTAMP => {
                let mut all_values = Vec::with_capacity(self.row_count);
                let mut all_nulls = Vec::with_capacity(self.row_count);
                for (gi, block) in col_blocks.iter().enumerate() {
                    self.decompress_block_into(
                        col_idx,
                        gi,
                        block,
                        type_tag,
                        num_groups,
                        &mut lz4_buf,
                        &mut all_nulls,
                        Some(&mut all_values),
                        None,
                        None,
                        None,
                        None,
                        None,
                    )?;
                }
                ColumnData::TimestampNanos {
                    values: all_values,
                    nulls: all_nulls,
                }
            }
            super::format::COL_BOOLEAN => {
                let mut all_values = Vec::with_capacity(self.row_count);
                let mut all_nulls = Vec::with_capacity(self.row_count);
                for (gi, block) in col_blocks.iter().enumerate() {
                    self.decompress_block_into(
                        col_idx,
                        gi,
                        block,
                        type_tag,
                        num_groups,
                        &mut lz4_buf,
                        &mut all_nulls,
                        None,
                        None,
                        None,
                        Some(&mut all_values),
                        None,
                        None,
                    )?;
                }
                ColumnData::Boolean {
                    values: all_values,
                    nulls: all_nulls,
                }
            }
            COL_DICTIONARY => {
                let mut all_ids = Vec::with_capacity(self.row_count);
                let mut all_nulls = Vec::with_capacity(self.row_count);
                for (gi, block) in col_blocks.iter().enumerate() {
                    self.decompress_block_into(
                        col_idx,
                        gi,
                        block,
                        type_tag,
                        num_groups,
                        &mut lz4_buf,
                        &mut all_nulls,
                        None,
                        None,
                        Some(&mut all_ids),
                        None,
                        None,
                        None,
                    )?;
                }
                let dictionary = dict.unwrap_or_else(|| Arc::from(Vec::<SmartString>::new()));
                // One pass over the ids clears a valid column; the null-aware
                // scan runs only when some id is out of range
                let out_of_range = all_ids
                    .iter()
                    .copied()
                    .max()
                    .is_some_and(|max_id| max_id as usize >= dictionary.len());
                if out_of_range
                    && all_ids
                        .iter()
                        .zip(&all_nulls)
                        .any(|(&id, &is_null)| !is_null && id as usize >= dictionary.len())
                {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "dictionary id out of range",
                    ));
                }
                ColumnData::Dictionary {
                    ids: all_ids,
                    dictionary,
                    nulls: all_nulls,
                }
            }
            COL_BYTES => {
                let mut all_data = Vec::new();
                let mut all_offsets = Vec::with_capacity(self.row_count);
                let mut all_nulls = Vec::with_capacity(self.row_count);
                for (gi, block) in col_blocks.iter().enumerate() {
                    self.decompress_block_into(
                        col_idx,
                        gi,
                        block,
                        type_tag,
                        num_groups,
                        &mut lz4_buf,
                        &mut all_nulls,
                        None,
                        None,
                        None,
                        None,
                        Some(&mut all_data),
                        Some(&mut all_offsets),
                    )?;
                }
                ColumnData::Bytes {
                    data: all_data,
                    offsets: all_offsets,
                    ext_type,
                    nulls: all_nulls,
                }
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unknown column type tag",
                ));
            }
        })
    }

    fn block_layout(
        &self,
        col_idx: usize,
        gi: usize,
        type_tag: u8,
        num_groups: usize,
    ) -> std::io::Result<(usize, usize)> {
        let decomp_len = *self
            .decompressed_lens
            .get(col_idx)
            .and_then(|lens| lens.get(gi))
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "missing block length")
            })?;
        let group_rows = self.group_row_count(gi, num_groups)?;
        let row_bytes = match type_tag {
            super::format::COL_INT64
            | super::format::COL_FLOAT64
            | super::format::COL_TIMESTAMP => 9,
            super::format::COL_BOOLEAN => 2,
            COL_DICTIONARY => 5,
            COL_BYTES => 0,
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "unknown column type tag",
                ));
            }
        };
        if (row_bytes != 0 && group_rows.checked_mul(row_bytes) != Some(decomp_len))
            || decomp_len > isize::MAX as usize
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid column block length",
            ));
        }
        if type_tag == COL_BYTES
            && group_rows
                .checked_mul(17)
                .and_then(|len| len.checked_add(16))
                .is_none_or(|len| len > decomp_len)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid bytes block length",
            ));
        }
        Ok((decomp_len, group_rows))
    }

    /// Decompress and deserialize a single block with context in error messages.
    #[allow(clippy::too_many_arguments)]
    fn decompress_block(
        &self,
        col_idx: usize,
        gi: usize,
        block: &[u8],
        type_tag: u8,
        num_groups: usize,
        dict: Option<Arc<[SmartString]>>,
        ext_type: DataType,
    ) -> std::io::Result<ColumnData> {
        let (decomp_len, group_rows) = self.block_layout(col_idx, gi, type_tag, num_groups)?;
        let raw_bytes = if block.len() == decomp_len {
            return deserialize_column_block(block, type_tag, group_rows, dict, ext_type);
        } else {
            let mut raw = Vec::new();
            raw.try_reserve_exact(decomp_len)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, e))?;
            raw.resize(decomp_len, 0);
            let decoded_len = lz4_flex::decompress_into(block, &mut raw).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "corrupt V4 block: col={}, group={}/{}: {}",
                        col_idx, gi, num_groups, e
                    ),
                )
            })?;
            if decoded_len != decomp_len {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "LZ4 decoded length does not match block length",
                ));
            }
            raw
        };
        deserialize_column_block(&raw_bytes, type_tag, group_rows, dict, ext_type)
    }

    /// Decompress and deserialize a single block, appending directly into
    /// the caller's output buffers. No intermediate `ColumnData` is created.
    /// `lz4_buf` is a reusable scratch buffer for LZ4 decompression — resized
    /// as needed but never freed between groups, eliminating per-group allocs.
    #[allow(clippy::too_many_arguments)]
    fn decompress_block_into(
        &self,
        col_idx: usize,
        gi: usize,
        block: &[u8],
        type_tag: u8,
        num_groups: usize,
        lz4_buf: &mut Vec<u8>,
        nulls_out: &mut Vec<bool>,
        i64_out: Option<&mut Vec<i64>>,
        f64_out: Option<&mut Vec<f64>>,
        u32_out: Option<&mut Vec<u32>>,
        bool_out: Option<&mut Vec<bool>>,
        bytes_data_out: Option<&mut Vec<u8>>,
        bytes_offsets_out: Option<&mut Vec<(u64, u64)>>,
    ) -> std::io::Result<()> {
        let (decomp_len, group_rows) = self.block_layout(col_idx, gi, type_tag, num_groups)?;
        if block.len() == decomp_len {
            return deserialize_column_block_into(
                block,
                type_tag,
                group_rows,
                nulls_out,
                i64_out,
                f64_out,
                u32_out,
                bool_out,
                bytes_data_out,
                bytes_offsets_out,
            );
        }
        // Reuse caller's LZ4 scratch buffer (grows once, reused across groups).
        if lz4_buf.len() < decomp_len {
            lz4_buf.resize(decomp_len, 0);
        }
        let decoded_len =
            lz4_flex::decompress_into(block, &mut lz4_buf[..decomp_len]).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "corrupt V4 block: col={}, group={}/{}: {}",
                        col_idx, gi, num_groups, e
                    ),
                )
            })?;
        if decoded_len != decomp_len {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "LZ4 decoded length does not match block length",
            ));
        }
        deserialize_column_block_into(
            &lz4_buf[..decomp_len],
            type_tag,
            group_rows,
            nulls_out,
            i64_out,
            f64_out,
            u32_out,
            bool_out,
            bytes_data_out,
            bytes_offsets_out,
        )
    }

    /// Decompress a single row group for one column. Returns the ColumnData
    /// covering only that group's rows (0..group_row_count).
    pub fn decompress_single_group(
        &self,
        col_idx: usize,
        group_idx: usize,
    ) -> std::io::Result<ColumnData> {
        let col_blocks = self.blocks.get(col_idx).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "column index out of range",
            )
        })?;
        let block = col_blocks.get(group_idx).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "group index out of range")
        })?;
        let num_groups = col_blocks.len();
        let type_tag = *self.col_type_tags.get(col_idx).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing column type tag")
        })?;
        let ext_type = self
            .col_ext_types
            .get(col_idx)
            .and_then(|&tag| DataType::from_u8(tag))
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid extension type tag",
                )
            })?;
        let dict: Option<Arc<[SmartString]>> = if type_tag == COL_DICTIONARY {
            self.col_dicts
                .iter()
                .find(|(ci, _)| *ci == col_idx)
                .map(|(_, arc)| Arc::clone(arc))
        } else {
            None
        };
        self.decompress_block(
            col_idx, group_idx, block, type_tag, num_groups, dict, ext_type,
        )
    }

    /// Look up a string in a Dictionary column's shared dictionary.
    /// Returns the dict_id without decompressing any column blocks.
    pub fn dict_lookup(&self, col_idx: usize, value: &str) -> Option<u32> {
        if self.col_type_tags[col_idx] != COL_DICTIONARY {
            return None;
        }
        let dict = self.col_dicts.iter().find(|(ci, _)| *ci == col_idx)?;
        for (i, s) in dict.1.iter().enumerate() {
            if s.as_str() == value {
                return Some(i as u32);
            }
        }
        None
    }

    /// Binary search on a sorted column using row-group zone maps.
    /// Decompresses only the group(s) containing the target value.
    /// Returns a proven global row index, or None when optional group metadata
    /// cannot support narrowing. Invalid geometry and decode failures propagate.
    pub fn binary_search_ge(
        &self,
        col_idx: usize,
        target: i64,
        row_groups: &[super::column::RowGroupMeta],
    ) -> std::io::Result<Option<usize>> {
        self.binary_search_impl(col_idx, target, row_groups, false)
    }

    pub fn binary_search_gt(
        &self,
        col_idx: usize,
        target: i64,
        row_groups: &[super::column::RowGroupMeta],
    ) -> std::io::Result<Option<usize>> {
        self.binary_search_impl(col_idx, target, row_groups, true)
    }

    fn binary_search_impl(
        &self,
        col_idx: usize,
        target: i64,
        row_groups: &[super::column::RowGroupMeta],
        strict: bool,
    ) -> std::io::Result<Option<usize>> {
        let num_groups = self
            .blocks
            .get(col_idx)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "column index out of range",
                )
            })?
            .len();
        match self.col_type_tags.get(col_idx).copied() {
            Some(super::format::COL_INT64 | super::format::COL_TIMESTAMP) => {}
            Some(
                super::format::COL_FLOAT64
                | super::format::COL_BOOLEAN
                | COL_DICTIONARY
                | COL_BYTES,
            ) => {
                return Ok(None);
            }
            _ => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid column type tag",
                ))
            }
        }
        if self.group_size == 0 || num_groups != self.row_count.div_ceil(self.group_size) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid row group geometry",
            ));
        }
        if num_groups == 0 {
            return Ok(Some(0));
        }

        if !row_groups.is_empty() {
            if row_groups.len() != num_groups {
                return Ok(None);
            }
            for (gi, rg) in row_groups.iter().enumerate() {
                let start = gi * self.group_size;
                let group_rows = (self.row_count - start).min(self.group_size);
                if rg.start_idx as usize != start || rg.end_idx as usize != start + group_rows {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "invalid row group bounds",
                    ));
                }
                let Some(zm) = rg.zone_maps.get(col_idx) else {
                    return Ok(None);
                };
                let max_i64 = match &zm.max {
                    crate::core::Value::Integer(v) => *v,
                    crate::core::Value::Timestamp(ts) => {
                        ts.timestamp_nanos_opt().unwrap_or(i64::MAX)
                    }
                    _ => return Ok(None),
                };
                if target > max_i64 {
                    continue;
                }
                let col = self.group_column(col_idx, gi)?;
                let local = if strict {
                    col.binary_search_gt(target)
                } else {
                    col.binary_search_ge(target)
                };
                if local < group_rows {
                    return Ok(Some(start + local));
                }
            }
            return Ok(Some(self.row_count));
        }

        if num_groups == 1 {
            let col = self.group_column(col_idx, 0)?;
            return Ok(Some(if strict {
                col.binary_search_gt(target)
            } else {
                col.binary_search_ge(target)
            }));
        }

        Ok(None)
    }

    /// Number of groups for a given column.
    pub fn num_groups(&self, col_idx: usize) -> usize {
        self.blocks[col_idx].len()
    }

    /// Number of rows in a specific group.
    fn group_row_count(&self, group_idx: usize, num_groups: usize) -> std::io::Result<usize> {
        if self.group_size == 0
            || num_groups != self.row_count.div_ceil(self.group_size)
            || group_idx >= num_groups
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid row group geometry",
            ));
        }
        Ok((self.row_count - group_idx * self.group_size).min(self.group_size))
    }

    /// Number of columns.
    pub fn col_count(&self) -> usize {
        self.blocks.len()
    }

    /// Return the shared dictionary Arc for a dictionary-encoded column.
    /// Returns None if the column is not dictionary-encoded.
    /// This avoids decompressing any column blocks.
    pub fn get_column_dictionary(&self, col_idx: usize) -> Option<&Arc<[SmartString]>> {
        if col_idx >= self.col_type_tags.len() || self.col_type_tags[col_idx] != COL_DICTIONARY {
            return None;
        }
        self.col_dicts
            .iter()
            .find(|(ci, _)| *ci == col_idx)
            .map(|(_, arc)| arc)
    }

    /// Total compressed bytes in RAM.
    pub fn memory_size(&self) -> usize {
        let mut size = 0;
        for col_blocks in &self.blocks {
            for block in col_blocks {
                size += block.len();
            }
        }
        // Add dictionary memory (col_dicts Arcs)
        for (_, dict) in &self.col_dicts {
            for s in dict.iter() {
                size += s.len() + 24; // SmartString overhead
            }
        }
        size
    }

    /// Access raw compressed blocks (for V4 write without re-compression).
    pub fn raw_blocks(&self) -> &[Vec<Vec<u8>>] {
        &self.blocks
    }

    /// Column type tags.
    pub fn col_type_tags(&self) -> &[u8] {
        &self.col_type_tags
    }

    /// Column ext types.
    pub fn col_ext_types(&self) -> &[u8] {
        &self.col_ext_types
    }

    /// Group size.
    pub fn group_size(&self) -> usize {
        self.group_size
    }

    /// Decompressed sizes per block.
    pub fn decompressed_lens(&self) -> &[Vec<usize>] {
        &self.decompressed_lens
    }
}

// =============================================================================
// LazyColumns: cached, fallible whole-column access
// =============================================================================

/// Column storage that decompresses from CompressedBlockStore on first access.
/// After OnceLock init, subsequent access is a pointer dereference (free).
pub struct LazyColumns {
    /// Per-column OnceLock slots. Empty until first access.
    slots: Vec<OnceLock<std::io::Result<ColumnData>>>,
    /// Compressed backing store. None for eagerly-loaded columns.
    /// Wrapped in Arc so warm-tier volumes can share the store cheaply.
    compressed_store: Option<Arc<CompressedBlockStore>>,
    /// Column data types (available without decompressing).
    col_data_types: Vec<DataType>,
    /// True when all OnceLock slots are initialized. Starts true for eager,
    /// false for deferred. Flipped to true when all OnceLock slots are
    /// populated (checked on slow path), so scanners skip per-group decompression.
    is_eager: std::sync::atomic::AtomicBool,
}

impl LazyColumns {
    /// Create from pre-loaded columns (VolumeBuilder::finish(), V4 eager load).
    /// All OnceLock slots are pre-initialized. No compressed store.
    pub fn eager(columns: Vec<ColumnData>, col_data_types: Vec<DataType>) -> Self {
        let slots = columns
            .into_iter()
            .map(|col| {
                let cell = OnceLock::new();
                let _ = cell.set(Ok(col));
                cell
            })
            .collect();
        Self {
            slots,
            compressed_store: None,
            col_data_types,
            is_eager: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Create with empty slots backed by a CompressedBlockStore.
    /// Columns are decompressed from RAM on first access (~4 GB/s).
    pub fn deferred(store: CompressedBlockStore, col_data_types: Vec<DataType>) -> Self {
        let col_count = store.col_count();
        let slots = (0..col_count).map(|_| OnceLock::new()).collect();
        Self {
            slots,
            compressed_store: Some(Arc::new(store)),
            col_data_types,
            is_eager: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create with empty slots backed by a shared CompressedBlockStore Arc.
    /// Used by warm-tier volumes to share the store cheaply.
    pub fn deferred_shared(
        store: Arc<CompressedBlockStore>,
        col_data_types: Vec<DataType>,
    ) -> Self {
        let col_count = store.col_count();
        let slots = (0..col_count).map(|_| OnceLock::new()).collect();
        Self {
            slots,
            compressed_store: Some(store),
            col_data_types,
            is_eager: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create columns with only data types (for cold-tier volumes).
    /// The volume must be reloaded from disk before column access.
    pub fn metadata_only(col_data_types: Vec<DataType>) -> Self {
        let col_count = col_data_types.len();
        Self {
            slots: (0..col_count).map(|_| OnceLock::new()).collect(),
            compressed_store: None,
            col_data_types,
            is_eager: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create empty LazyColumns (for Scanner::empty()).
    pub fn empty() -> Self {
        Self {
            slots: Vec::new(),
            compressed_store: None,
            col_data_types: Vec::new(),
            is_eager: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// Attach a compressed backing store to an eager LazyColumns.
    /// The store enables warm-tier eviction: decompressed columns can be
    /// dropped and re-decompressed from the in-memory compressed blocks.
    /// Does not change is_eager (scan path remains OnceLock-based).
    pub fn attach_compressed_store(&mut self, store: CompressedBlockStore) {
        self.compressed_store = Some(Arc::new(store));
    }

    /// Whether all OnceLock slots are initialized (eager mode).
    pub fn is_eager(&self) -> bool {
        self.is_eager.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Number of columns.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether there are no columns.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Get the DataType for a column without decompressing it.
    #[inline]
    pub fn data_type(&self, idx: usize) -> DataType {
        self.col_data_types[idx]
    }

    /// Iterator over all columns (triggers decompression of unloaded columns).
    pub fn iter(&self) -> LazyColumnsIter<'_> {
        LazyColumnsIter {
            columns: self,
            idx: 0,
        }
    }

    /// Estimate in-memory size: compressed store + loaded columns.
    pub fn memory_size(&self) -> usize {
        let mut size = 0;
        // Compressed store
        if let Some(ref store) = self.compressed_store {
            size += store.memory_size();
        }
        // Loaded (decompressed) columns
        for slot in &self.slots {
            if let Some(Ok(col)) = slot.get() {
                size += col.memory_size();
            }
        }
        size
    }

    /// Whether this LazyColumns has a compressed backing store.
    pub fn has_compressed_store(&self) -> bool {
        self.compressed_store.is_some()
    }

    /// Whether the scanner should use per-group decompression from the
    /// CompressedBlockStore. Returns false when all columns are already
    /// loaded in OnceLock slots (eager after seal/compaction),
    /// because direct OnceLock access is faster than re-decompressing groups.
    #[inline]
    pub fn should_use_group_cache(&self) -> bool {
        !self.is_eager.load(std::sync::atomic::Ordering::Relaxed) && self.compressed_store.is_some()
    }

    /// Return the dictionary for a dictionary-encoded column without
    /// decompressing column data. Checks loaded OnceLock slots first,
    /// then falls back to the CompressedBlockStore's pre-built dictionary.
    /// Returns None for non-dictionary columns and propagates cached failures.
    pub fn get_column_dictionary(
        &self,
        col_idx: usize,
    ) -> std::io::Result<Option<Arc<[SmartString]>>> {
        let slot = self.slots.get(col_idx).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "column index out of range",
            )
        })?;
        // Fast path: column already loaded in OnceLock
        if let Some(col) = slot.get() {
            let col = col
                .as_ref()
                .map_err(|e| std::io::Error::new(e.kind(), e.to_string()))?;
            if let ColumnData::Dictionary { dictionary, .. } = col {
                return Ok(Some(Arc::clone(dictionary)));
            }
            return Ok(None);
        }
        // Slow path: extract from compressed store without decompressing
        let store = self.compressed_store.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "column data is not loaded")
        })?;
        Ok(store.get_column_dictionary(col_idx).cloned())
    }

    /// Access the compressed store (for V4 write).
    pub fn compressed_store(&self) -> Option<&CompressedBlockStore> {
        self.compressed_store.as_ref().map(|a| a.as_ref())
    }

    /// Access the compressed store as a shared Arc (for warm-tier cloning).
    pub fn compressed_store_arc(&self) -> Option<&Arc<CompressedBlockStore>> {
        self.compressed_store.as_ref()
    }

    /// Decode and take ownership of every column, consuming the LazyColumns.
    pub fn take_columns(self) -> std::io::Result<Vec<ColumnData>> {
        for idx in 0..self.len() {
            self.get(idx)?;
        }
        let mut result = Vec::with_capacity(self.slots.len());
        for slot in self.slots {
            if let Some(col) = slot.into_inner() {
                result.push(col?);
            }
        }
        Ok(result)
    }

    /// Borrow a column, caching both successful decodes and failures.
    #[inline]
    pub fn get(&self, idx: usize) -> std::io::Result<&ColumnData> {
        let slot = self.slots.get(idx).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "column index out of range",
            )
        })?;
        // Fast path: already initialized
        if let Some(Ok(col)) = slot.get() {
            return Ok(col);
        }
        self.load_column(idx)
    }

    #[cold]
    fn load_column(&self, idx: usize) -> std::io::Result<&ColumnData> {
        let col = self.slots[idx].get_or_init(|| {
            let store = self.compressed_store.as_ref().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "column data is not loaded")
            })?;
            store.decompress_column(idx)
        });
        // Promote only after every column has decoded successfully.
        if !self.is_eager.load(std::sync::atomic::Ordering::Relaxed)
            && self.slots.iter().all(|s| matches!(s.get(), Some(Ok(_))))
        {
            self.is_eager
                .store(true, std::sync::atomic::Ordering::Relaxed);
        }
        col.as_ref()
            .map_err(|e| std::io::Error::new(e.kind(), e.to_string()))
    }
}

/// Iterator over LazyColumns that triggers decompression on access.
pub struct LazyColumnsIter<'a> {
    columns: &'a LazyColumns,
    idx: usize,
}

impl<'a> Iterator for LazyColumnsIter<'a> {
    type Item = std::io::Result<&'a ColumnData>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx < self.columns.len() {
            let col = self.columns.get(self.idx);
            self.idx += 1;
            Some(col)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.columns.len() - self.idx;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for LazyColumnsIter<'_> {}

impl<'a> IntoIterator for &'a LazyColumns {
    type Item = std::io::Result<&'a ColumnData>;
    type IntoIter = LazyColumnsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Immutable metadata shared across hot/warm/cold volume tiers.
/// Wrapped in Arc so eviction only swaps LazyColumns, zero allocation.
#[derive(Clone)]
pub struct VolumeMeta {
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
    /// Precomputed lowercase column name -> index map for O(1) lookup.
    /// Built once at construction; replaces O(C) linear scan in column_index().
    pub column_name_map: AHashMap<SmartString, usize>,
    /// Row group metadata for sub-volume zone map pruning.
    /// Empty for volumes with <= ROW_GROUP_SIZE rows (single implicit group).
    pub row_groups: Vec<super::column::RowGroupMeta>,
}

impl VolumeMeta {
    /// Estimate the in-memory size of this metadata in bytes.
    pub fn memory_size(&self) -> usize {
        let mut size = 0usize;
        // row_ids: Vec<i64>
        size += self.row_ids.len() * 8;
        // zone_maps: 2 Values (16 bytes each) + 2 u32 per column
        size += self.zone_maps.len() * (16 + 16 + 8);
        // bloom_filters: Vec<u64> bitsets
        for bf in &self.bloom_filters {
            size += bf.memory_size();
        }
        // stats: 16 bytes base + per-column (i128 + f64 + u64 + 2 Values + u64)
        size += 16 + self.stats.columns.len() * (16 + 8 + 8 + 16 + 16 + 8);
        // column_names
        for name in &self.column_names {
            size += name.len() + 24;
        }
        // column_types + sorted_columns
        size += self.column_types.len() * 2;
        size += self.sorted_columns.len();
        // column_name_map: ~72 bytes per entry (SmartString + usize + hash overhead)
        size += self.column_name_map.len() * 72;
        // row_groups: per group has start/end u32 + Vec<ZoneMap>
        for rg in &self.row_groups {
            size += 8 + rg.zone_maps.len() * (16 + 16 + 8);
        }
        size
    }
}

/// A frozen volume ready for queries.
///
/// This is the in-memory representation. Serialization to/from disk
/// is handled by io.rs (V4 format).
pub struct FrozenVolume {
    /// Column data stored as typed arrays with lazy decompression
    pub columns: LazyColumns,
    /// Shared metadata (zone maps, bloom filters, stats, row IDs, etc.)
    pub meta: Arc<VolumeMeta>,
    /// Per-volume unique index: lazily built, never invalidated (volume is immutable).
    /// Key: sorted column indices for a UNIQUE constraint.
    /// Value: shared sorted (hash, row_idx) pairs, 16 bytes per entry.
    #[allow(clippy::type_complexity)]
    pub unique_indices:
        Arc<parking_lot::RwLock<rustc_hash::FxHashMap<Vec<usize>, Arc<Vec<(u64, u32)>>>>>,
    /// Access epoch counter. Bumped per scan for eviction tracking.
    pub last_access_epoch: std::sync::atomic::AtomicU64,
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

            // Skip NaN for both stats and zone maps — NaN poisons sum_float
            // (NaN + x = NaN) and corrupts zm.max (compare_floats treats NaN
            // as greater-than-all). Row-group zone maps handle NaN separately.
            let is_nan = matches!(value, Value::Float(f) if f.is_nan());
            if !is_nan {
                self.stats.columns[col_idx].accumulate(value);
            }

            if !is_nan {
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
                        Value::Extension(data) if data.len() > 1 => {
                            &data[1..] // skip type tag
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
        debug_assert_eq!(self.row_ids.len(), self.row_count);
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
                    dictionary: Arc::from(std::mem::take(&mut self.dict_tables[idx])),
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
                            // Extension types (JSON, Vector) all hash to tag 0
                            // in hash_value (no payload). Insert the same no-op
                            // hash directly — avoids constructing
                            // Value::Extension(CompactArc) per cell.
                            bf.add_extension_noop();
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
            .flat_map(|(i, name)| {
                let lower = SmartString::from(name.to_lowercase());
                let original = SmartString::from(name.as_str());
                if lower == original {
                    // Already lowercase — one entry
                    vec![(lower, i)]
                } else {
                    // Store both original and lowercase for zero-alloc lookup
                    vec![(original, i), (lower, i)]
                }
            })
            .collect();

        // Build row-group zone maps for sub-volume pruning.
        // Only worth it for volumes larger than one group.
        let row_groups = if self.row_count > super::column::ROW_GROUP_SIZE {
            let mut groups = Vec::new();
            let mut start = 0;
            while start < self.row_count {
                let end = (start + super::column::ROW_GROUP_SIZE).min(self.row_count);
                let group_zone_maps: Vec<super::column::ZoneMap> = columns
                    .iter()
                    .map(|col| col.zone_map_for_range(start, end))
                    .collect();
                groups.push(super::column::RowGroupMeta {
                    start_idx: start as u32,
                    end_idx: end as u32,
                    zone_maps: group_zone_maps,
                });
                start = end;
            }
            groups
        } else {
            Vec::new()
        };

        FrozenVolume {
            columns: LazyColumns::eager(columns, column_types.clone()),
            meta: Arc::new(VolumeMeta {
                zone_maps: self.zone_maps,
                bloom_filters,
                stats: self.stats,
                row_count: self.row_count,
                column_names,
                column_types,
                row_ids: self.row_ids,
                sorted_columns,
                column_name_map,
                row_groups,
            }),
            unique_indices: Arc::new(parking_lot::RwLock::new(rustc_hash::FxHashMap::default())),
            last_access_epoch: std::sync::atomic::AtomicU64::new(
                GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

/// Source for a single schema column when reading from a frozen volume.
/// Precomputed once per volume per scan, then used for every row.
#[derive(Clone)]
pub enum ColSource {
    /// Schema column maps to this volume column index.
    Volume(usize),
    /// Schema column was added after this volume was sealed.
    /// Use this default value (NULL or DEFAULT from ALTER TABLE).
    Default(Value),
}

/// Precomputed mapping from current schema to a frozen volume's columns.
/// Computed once per volume per scan. Eliminates per-row name lookups.
#[derive(Clone)]
pub struct ColumnMapping {
    /// For each schema column position, how to get the value.
    pub sources: Vec<ColSource>,
    /// True when every schema column maps 1:1 to the same volume column
    /// in the same order. When true, callers can skip the mapping and
    /// use get_row()/get_row_projected() directly.
    pub is_identity: bool,
    _payload: PayloadCharge,
    pub(crate) exports: DeferredExports,
}

impl ColumnMapping {
    pub fn new(sources: Vec<ColSource>, is_identity: bool) -> Self {
        let bytes = sources
            .iter()
            .map(|source| match source {
                ColSource::Default(value) => value.heap_bytes() as u128,
                ColSource::Volume(_) => 0,
            })
            .sum();
        Self {
            sources,
            is_identity,
            _payload: PayloadCharge::unshared(bytes),
            exports: DeferredExports::default(),
        }
    }
}

/// Compute column mapping from current schema to a frozen volume.
/// Handles renames (via column_renames fallback) and drops.
/// `volume_schema_version` is the schema epoch when the volume was created.
/// For dropped columns, only volumes created before or at the drop are masked.
/// Callers should use SegmentManager::get_volume_mapping() which caches the result.
pub fn compute_column_mapping_with_drops(
    schema: &Schema,
    volume: &FrozenVolume,
    dropped_columns: &[(crate::common::SmartString, u64)],
    volume_schema_version: u64,
    column_renames: &[(crate::common::SmartString, crate::common::SmartString)],
) -> ColumnMapping {
    let mut sources = Vec::with_capacity(schema.columns.len());
    let mut is_identity = schema.columns.len() == volume.columns.len();
    // Track which volume column indices are already claimed. Prevents two
    // schema columns from binding to the same physical column (e.g., after
    // RENAME a→b then ADD COLUMN a, both "b" via rename and "a" via direct
    // match would hit the same old physical column without this guard).
    let mut used_vol_indices = smallvec::SmallVec::<[usize; 16]>::new();

    for (pos, col) in schema.columns.iter().enumerate() {
        // Try rename fallback FIRST (higher priority: a renamed column's
        // old physical slot belongs to the renamed column, not a new column
        // that happens to reuse the old name).
        let vol_idx = column_renames
            .iter()
            .find(|(_, new)| new.as_str() == col.name_lower)
            .and_then(|(old, _)| volume.column_index(old.as_str()))
            .or_else(|| volume.column_index(&col.name_lower));

        if let Some(vol_idx) = vol_idx {
            let type_matches = vol_idx < volume.meta.column_types.len()
                && volume.meta.column_types[vol_idx] == col.data_type;
            let was_dropped = dropped_columns.iter().any(|(d, drop_ver)| {
                d.as_str() == col.name_lower && volume_schema_version <= *drop_ver
            });
            let already_used = used_vol_indices.contains(&vol_idx);
            if type_matches && !was_dropped && !already_used {
                if is_identity && vol_idx != pos {
                    is_identity = false;
                }
                used_vol_indices.push(vol_idx);
                sources.push(ColSource::Volume(vol_idx));
            } else {
                is_identity = false;
                if let Some(ref default_val) = col.default_value {
                    sources.push(ColSource::Default(default_val.clone()));
                } else {
                    sources.push(ColSource::Default(Value::Null(col.data_type)));
                }
            }
        } else {
            // Column not in volume (added after seal, or dropped+re-added)
            is_identity = false;
            if let Some(ref default_val) = col.default_value {
                sources.push(ColSource::Default(default_val.clone()));
            } else {
                sources.push(ColSource::Default(Value::Null(col.data_type)));
            }
        }
    }

    ColumnMapping::new(sources, is_identity)
}

impl FrozenVolume {
    /// Borrow the resident physical row IDs.
    #[inline]
    pub fn row_ids(&self) -> std::io::Result<&[i64]> {
        Ok(&self.meta.row_ids)
    }

    /// Get a row using a precomputed column mapping.
    /// Materializes all schema columns through the mapping.
    pub fn get_row_mapped(&self, idx: usize, mapping: &ColumnMapping) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(mapping.sources.len());
        let mut exported = 0;
        for src in &mapping.sources {
            values.push(match src {
                ColSource::Volume(vol_idx) => self.columns.get(*vol_idx)?.get_value(idx),
                ColSource::Default(val) => {
                    exported += val.heap_bytes() as u128;
                    val.clone()
                }
            });
        }
        mapping.exports.add(exported);
        Ok(Row::from_values(values))
    }

    /// Get specific columns of a row using a precomputed column mapping.
    /// Only materializes the requested schema columns — skips the rest.
    pub fn get_row_mapped_projected(
        &self,
        idx: usize,
        mapping: &ColumnMapping,
        col_indices: &[usize],
    ) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(col_indices.len());
        let mut exported = 0;
        for &ci in col_indices {
            values.push(match &mapping.sources[ci] {
                ColSource::Volume(vol_idx) => self.columns.get(*vol_idx)?.get_value(idx),
                ColSource::Default(val) => {
                    exported += val.heap_bytes() as u128;
                    val.clone()
                }
            });
        }
        mapping.exports.add(exported);
        Ok(Row::from_values(values))
    }

    /// Get a row materializing only columns marked true in the mask.
    /// Other columns get typed Null (stack-only, zero allocation).
    /// The row has full schema width so filter column indices work.
    /// Uses LazyColumns::data_type() for unneeded columns to avoid decompression.
    #[inline]
    pub fn get_row_needed(&self, idx: usize, needed: &[bool]) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(self.columns.len());
        for ci in 0..self.columns.len() {
            values.push(if ci < needed.len() && needed[ci] {
                self.columns.get(ci)?.get_value(idx)
            } else {
                Value::Null(self.columns.data_type(ci))
            });
        }
        Ok(Row::from_values(values))
    }

    /// Get a row using a mapping, materializing only needed columns.
    /// Combines schema evolution (mapping) with column pruning (mask).
    /// Uses LazyColumns::data_type() for unneeded columns to avoid decompression.
    #[inline]
    pub fn get_row_mapped_needed(
        &self,
        idx: usize,
        mapping: &ColumnMapping,
        needed: &[bool],
    ) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(mapping.sources.len());
        let mut exported = 0;
        for (ci, src) in mapping.sources.iter().enumerate() {
            values.push(if ci < needed.len() && needed[ci] {
                match src {
                    ColSource::Volume(vol_idx) => self.columns.get(*vol_idx)?.get_value(idx),
                    ColSource::Default(val) => {
                        exported += val.heap_bytes() as u128;
                        val.clone()
                    }
                }
            } else {
                match src {
                    ColSource::Volume(vol_idx) => Value::Null(self.columns.data_type(*vol_idx)),
                    ColSource::Default(val) => Value::Null(val.data_type()),
                }
            });
        }
        mapping.exports.add(exported);
        Ok(Row::from_values(values))
    }

    /// Get a row as a Vec of Values (for executor compatibility).
    pub fn get_row(&self, idx: usize) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(self.columns.len());
        for col in &self.columns {
            values.push(col?.get_value(idx));
        }
        Ok(Row::from_values(values))
    }

    /// Get specific columns of a row (projection pushdown).
    pub fn get_row_projected(&self, idx: usize, col_indices: &[usize]) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(col_indices.len());
        for &col in col_indices {
            values.push(self.columns.get(col)?.get_value(idx));
        }
        Ok(Row::from_values(values))
    }

    /// Check if a column is sorted (enables binary search).
    #[inline]
    pub fn is_sorted(&self, col_idx: usize) -> bool {
        self.meta.sorted_columns[col_idx]
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
    ) -> std::io::Result<()> {
        use std::hash::{Hash, Hasher};

        // Compute hash of query values
        let mut hasher = ahash::AHasher::default();
        for &val in values {
            val.hash(&mut hasher);
        }
        let hash = hasher.finish();

        let cached = {
            let indices = self.unique_indices.read();
            if let Some(entries) = indices.get(col_indices) {
                let pos = entries.partition_point(|&(h, _)| h < hash);
                if entries.get(pos).is_none_or(|&(h, _)| h != hash) {
                    return Ok(());
                }
                Some((Arc::clone(entries), pos))
            } else {
                None
            }
        };
        let (entries, pos) = match cached {
            Some(cached) => cached,
            None => {
                let entries = self.unique_index(col_indices)?;
                let pos = entries.partition_point(|&(h, _)| h < hash);
                (entries, pos)
            }
        };
        for &(h, row_idx) in &entries[pos..] {
            if h != hash {
                break;
            }
            let mut matches = true;
            for (&ci, &val) in col_indices.iter().zip(values) {
                let vol_val = self.columns.get(ci)?.get_value(row_idx as usize);
                if vol_val.is_null() || vol_val != *val {
                    matches = false;
                    break;
                }
            }
            if matches && f(row_idx) {
                break;
            }
        }

        Ok(())
    }

    /// Pre-build the unique sorted index for a set of column indices.
    /// Called during seal/compaction so the first INSERT after seal doesn't
    /// pay a ~60ms stall scanning all rows to build the index.
    pub fn prebuild_unique_index(&self, col_indices: &[usize]) -> std::io::Result<()> {
        self.unique_index(col_indices).map(|_| ())
    }

    fn unique_index(&self, col_indices: &[usize]) -> std::io::Result<Arc<Vec<(u64, u32)>>> {
        use std::hash::{Hash, Hasher};
        let cached = self.unique_indices.read().get(col_indices).cloned();
        if let Some(entries) = cached {
            return Ok(entries);
        }
        let columns = col_indices
            .iter()
            .map(|&ci| self.columns.get(ci))
            .collect::<std::io::Result<smallvec::SmallVec<[&ColumnData; 4]>>>()?;
        let mut entries: Vec<(u64, u32)> = Vec::with_capacity(self.meta.row_count);
        for row_idx in 0..self.meta.row_count {
            let mut row_hasher = ahash::AHasher::default();
            let mut has_null = false;
            for &col in &columns {
                if col.is_null(row_idx) {
                    has_null = true;
                    break;
                }
                col.get_value(row_idx).hash(&mut row_hasher);
            }
            if has_null {
                continue;
            }
            entries.push((row_hasher.finish(), row_idx as u32));
        }
        entries.sort_unstable_by_key(|&(h, _)| h);
        let entries = Arc::new(entries);
        let key = col_indices.to_vec();
        let replaced = self
            .unique_indices
            .write()
            .insert(key, Arc::clone(&entries));
        drop(replaced);
        Ok(entries)
    }

    /// Find the column index by name. O(1) via precomputed hashmap.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        if let Some(&idx) = self.meta.column_name_map.get(name) {
            return Some(idx);
        }
        let lower = name.to_lowercase();
        self.meta.column_name_map.get(lower.as_str()).copied()
    }

    /// Merge a column rename directly into column_name_map.
    /// Must be called BEFORE wrapping in Arc (takes &mut self).
    /// After this, column_index() finds both old and new names via the map.
    pub fn merge_column_rename(&mut self, new_name: &str, old_name: &str) {
        let meta = Arc::make_mut(&mut self.meta);
        let old_lower = SmartString::from(old_name.to_lowercase());
        let new_lower = SmartString::from(new_name.to_lowercase());
        if let Some(&idx) = meta.column_name_map.get(&old_lower) {
            meta.column_name_map.insert(new_lower, idx);
        } else if let Some(&idx) = meta.column_name_map.get(&new_lower) {
            // Already has the new name (chained rename handled)
            let _ = idx;
        }
    }

    /// Return the dictionary for a dictionary-encoded column.
    /// Avoids decompressing the full column — extracts from the shared
    /// dictionary stored in the compressed block store or OnceLock slot.
    /// Returns None for non-dictionary columns and propagates cached failures.
    pub fn get_column_dictionary(
        &self,
        col_idx: usize,
    ) -> std::io::Result<Option<Arc<[SmartString]>>> {
        self.columns.get_column_dictionary(col_idx)
    }

    /// Estimate the in-memory size of this volume in bytes.
    /// Counts metadata + compressed store + loaded (decompressed) columns.
    pub fn memory_size(&self) -> usize {
        self.meta.memory_size() + self.columns.memory_size()
    }

    /// Mark this volume as recently accessed. Stores u64::MAX as a sentinel
    /// meaning "accessed since last eviction cycle." The eviction pass resets
    /// non-evicted volumes to current_epoch, so the idle counter only starts
    /// after the last access.
    #[inline]
    pub fn mark_accessed(&self) {
        self.last_access_epoch
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether this volume is warm (compressed blocks in RAM, no decompressed columns).
    pub fn is_warm(&self) -> bool {
        !self.columns.is_eager() && self.columns.has_compressed_store()
    }

    /// Whether this volume is cold (no column data, no compressed blocks).
    pub fn is_cold(&self) -> bool {
        !self.columns.is_eager() && !self.columns.has_compressed_store()
    }

    /// Create a warm-tier volume: shares metadata via Arc (zero copy),
    /// shares compressed store via Arc, drops decompressed columns.
    /// Scanners use per-group decompression from RAM (~1ms per column per group).
    pub fn to_warm(&self) -> Option<FrozenVolume> {
        let store = self.columns.compressed_store_arc()?.clone();
        Some(FrozenVolume {
            columns: LazyColumns::deferred_shared(store, self.meta.column_types.clone()),
            meta: Arc::clone(&self.meta),
            unique_indices: Arc::clone(&self.unique_indices),
            // Start at current epoch so warm gets MIN_IDLE_CYCLES before cold.
            last_access_epoch: std::sync::atomic::AtomicU64::new(
                GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            ),
        })
    }

    /// Create a cold-tier volume: shares metadata via Arc (zero copy),
    /// drops both decompressed columns AND compressed blocks.
    /// Must reload from disk to scan.
    pub fn to_cold(&self) -> FrozenVolume {
        FrozenVolume {
            columns: LazyColumns::metadata_only(self.meta.column_types.clone()),
            meta: Arc::clone(&self.meta),
            unique_indices: Arc::clone(&self.unique_indices),
            // Start at current epoch so cold gets MIN_IDLE_CYCLES before
            // being considered for reload/re-eviction.
            last_access_epoch: std::sync::atomic::AtomicU64::new(
                GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SchemaBuilder;

    #[test]
    fn column_mapping_clones_keep_default_payload_charges() {
        let value = Value::text("retained mapped default ".repeat(256));
        let bytes = value.heap_bytes() as u128;
        let mapping = ColumnMapping::new(vec![ColSource::Default(value)], false);
        assert_eq!(mapping._payload.bytes(), bytes);
        let retained_mapping = mapping.clone();
        drop(mapping);
        assert_eq!(retained_mapping._payload.bytes(), bytes);
    }

    #[test]
    fn mapped_defaults_export_only_selected_children() {
        let schema = SchemaBuilder::new("mapped")
            .add_primary_key("id", DataType::Integer)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Integer(1)]));
        let volume = builder.finish();
        let value = Value::text("retained mapped default ".repeat(256));
        let bytes = value.heap_bytes() as u128;
        let mapping =
            ColumnMapping::new(vec![ColSource::Volume(0), ColSource::Default(value)], false);
        let retained_mapping = mapping.clone();
        drop(mapping);
        drop(schema);
        let scope = Arc::new(crate::storage::mvcc::read_memory::ReadScope::default());
        let _active = scope.enter();
        let full = volume.get_row_mapped(0, &retained_mapping).unwrap();
        let projected = volume
            .get_row_mapped_projected(0, &retained_mapping, &[1])
            .unwrap();
        let needed = volume
            .get_row_mapped_needed(0, &retained_mapping, &[false, true])
            .unwrap();
        volume
            .get_row_mapped_projected(0, &retained_mapping, &[0])
            .unwrap();
        volume
            .get_row_mapped_needed(0, &retained_mapping, &[true, false])
            .unwrap();
        drop(retained_mapping);
        assert_eq!(scope.exported_bytes(), 3 * bytes);
        assert_eq!(full.get(1), projected.get(0));
        assert_eq!(full.get(1), needed.get(1));
    }

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

        assert_eq!(volume.meta.row_count, 3);
        assert_eq!(volume.columns.len(), 4);
        assert_eq!(volume.meta.stats.count_star(), 3);

        // Check typed access
        assert_eq!(volume.columns.get(0).unwrap().get_i64(0), 1);
        assert_eq!(volume.columns.get(0).unwrap().get_i64(2), 3);
        assert_eq!(volume.columns.get(3).unwrap().get_f64(1), 101.5);

        // Check dictionary encoding
        assert_eq!(volume.columns.get(2).unwrap().get_str(0), "binance");
        assert_eq!(volume.columns.get(2).unwrap().get_str(1), "coinbase");
        assert_eq!(volume.columns.get(2).unwrap().get_str(2), "binance");
        // binance appears twice but uses same dict ID
        assert_eq!(
            volume.columns.get(2).unwrap().get_dict_id(0),
            volume.columns.get(2).unwrap().get_dict_id(2)
        );

        // Check zone maps
        assert_eq!(volume.meta.zone_maps[0].min, Value::Integer(1));
        assert_eq!(volume.meta.zone_maps[0].max, Value::Integer(3));
        assert_eq!(volume.meta.zone_maps[3].min, Value::Float(99.0));
        assert_eq!(volume.meta.zone_maps[3].max, Value::Float(101.5));

        // Check stats
        assert_eq!(volume.meta.stats.sum(3), 300.5); // 100.0 + 101.5 + 99.0

        // Check sortedness
        assert!(volume.is_sorted(0)); // id is sorted
        assert!(volume.is_sorted(1)); // time is sorted

        // Check row reconstruction
        let row = volume.get_row(0).unwrap();
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
        assert!(volume.columns.get(1).unwrap().is_null(0));
        assert!(volume.columns.get(3).unwrap().is_null(0));
        assert!(!volume.columns.get(0).unwrap().is_null(0));

        let row = volume.get_row(0).unwrap();
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
        let idx = volume
            .columns
            .get(0)
            .unwrap()
            .binary_search_ge(target_nanos);
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
        let row = volume.get_row_projected(0, &[0, 3]).unwrap();
        assert_eq!(row.len(), 2);
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(1), Some(&Value::Float(100.0)));
    }
}
