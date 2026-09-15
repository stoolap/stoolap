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
use crate::core::{DataType, Error, Result, Row, Schema, Value};

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
/// A volume file as every store that reads it shares it: the path a read
/// opens, which a table rename moves under every holder, and whether the
/// file is to be removed once the last holder lets go. One per file in
/// the process, so a volume reloaded into a new store shares the file
/// with the readers still holding the old one
pub struct VolumeFile {
    id: u64,
    path: parking_lot::RwLock<std::path::PathBuf>,
    retired: std::sync::atomic::AtomicBool,
}

static NEXT_FILE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The files of the process by path, and where a file whose last holder
/// is letting go is now, by the file's id
#[derive(Default)]
struct VolumeFiles {
    files: std::collections::HashMap<std::path::PathBuf, (u64, std::sync::Weak<VolumeFile>)>,
    letting_go: std::collections::HashMap<u64, std::path::PathBuf>,
}

static VOLUME_FILES: std::sync::LazyLock<parking_lot::Mutex<VolumeFiles>> =
    std::sync::LazyLock::new(|| parking_lot::Mutex::new(VolumeFiles::default()));

impl VolumeFile {
    /// The handle of the file at `path`, the one every holder shares
    pub fn shared(path: &std::path::Path) -> Arc<VolumeFile> {
        let mut registry = VOLUME_FILES.lock();
        if let Some(file) = registry
            .files
            .get(path)
            .and_then(|(_, file)| file.upgrade())
        {
            return file;
        }
        let file = Arc::new(VolumeFile {
            id: NEXT_FILE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            path: parking_lot::RwLock::new(path.to_path_buf()),
            retired: std::sync::atomic::AtomicBool::new(false),
        });
        registry
            .files
            .insert(path.to_path_buf(), (file.id, Arc::downgrade(&file)));
        file
    }

    /// Where the file is now
    pub fn path(&self) -> std::path::PathBuf {
        self.path.read().clone()
    }

    /// Removes the file once the last holder lets go
    pub fn retire(&self) {
        self.retired
            .store(true, std::sync::atomic::Ordering::Release);
    }

    fn open(&self) -> std::io::Result<std::fs::File> {
        std::fs::File::open(&*self.path.read())
    }

    /// Moves every file under `old_dir` to `new_dir` through `move_dir`:
    /// a holder's read meanwhile waits for the move and opens the new
    /// path after it; a failed move leaves every path as it was
    pub fn relocate(
        old_dir: &std::path::Path,
        new_dir: &std::path::Path,
        move_dir: impl FnOnce() -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        let mut registry = VOLUME_FILES.lock();
        let (result, held) = Self::relocate_in(&mut registry, old_dir, new_dir, move_dir);
        // Strong holders outlive the registry lock: the last one's drop takes it
        drop(registry);
        drop(held);
        result
    }

    /// The move under the registry lock; a file whose last holder is
    /// letting go, its drop waiting on the lock, moves too, its place
    /// kept by id so that drop removes it where it is
    fn relocate_in(
        registry: &mut VolumeFiles,
        old_dir: &std::path::Path,
        new_dir: &std::path::Path,
        move_dir: impl FnOnce() -> std::io::Result<()>,
    ) -> (std::io::Result<()>, Vec<Arc<VolumeFile>>) {
        let mut moved: Vec<(std::path::PathBuf, Arc<VolumeFile>)> = Vec::new();
        let mut letting_go: Vec<(std::path::PathBuf, u64)> = Vec::new();
        for (path, (id, file)) in registry.files.iter() {
            if !path.starts_with(old_dir) {
                continue;
            }
            match file.upgrade() {
                Some(file) => moved.push((path.clone(), file)),
                None => letting_go.push((path.clone(), *id)),
            }
        }
        let mut paths: Vec<_> = moved.iter().map(|(_, file)| file.path.write()).collect();
        let result = move_dir();
        if result.is_ok() {
            for ((old_path, file), path) in moved.iter().zip(paths.iter_mut()) {
                let Ok(rest) = old_path.strip_prefix(old_dir) else {
                    continue;
                };
                let new_path = new_dir.join(rest);
                registry.files.remove(old_path);
                registry
                    .files
                    .insert(new_path.clone(), (file.id, Arc::downgrade(file)));
                **path = new_path;
            }
            for (old_path, id) in letting_go {
                let Ok(rest) = old_path.strip_prefix(old_dir) else {
                    continue;
                };
                let new_path = new_dir.join(rest);
                if let Some(entry) = registry.files.remove(&old_path) {
                    registry.files.insert(new_path.clone(), entry);
                }
                registry.letting_go.insert(id, new_path);
            }
        }
        drop(paths);
        (result, moved.into_iter().map(|(_, file)| file).collect())
    }
}

impl Drop for VolumeFile {
    fn drop(&mut self) {
        // The file goes under the registry lock, which a move holds too
        let mut registry = VOLUME_FILES.lock();
        let path = registry
            .letting_go
            .remove(&self.id)
            .unwrap_or_else(|| self.path.get_mut().clone());
        if registry
            .files
            .get(&path)
            .is_some_and(|(_, file)| file.strong_count() == 0)
        {
            registry.files.remove(&path);
        }
        if self.retired.load(std::sync::atomic::Ordering::Acquire) {
            let _ = std::fs::remove_file(&path);
        }
        drop(registry);
    }
}

/// Where a store's blocks live: in memory for a volume just sealed or
/// compressed from its columns, or in the volume's file, read by position
/// when a group is decoded
enum BlockSource {
    Memory(Vec<Vec<Vec<u8>>>),
    /// The file is opened for each block read and closed after it, so a
    /// store holds no descriptor between reads
    File {
        file: Arc<VolumeFile>,
        /// offsets[col_idx][group_idx] and lens[col_idx][group_idx] of
        /// the compressed block in the file
        offsets: Vec<Vec<u64>>,
        lens: Vec<Vec<usize>>,
    },
}

/// Reads `buf.len()` bytes at `offset` without moving a shared cursor
fn read_exact_at(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "volume file ends inside a block",
                ));
            }
            done += n;
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positional volume reads are not supported on this platform",
        ))
    }
}

pub struct CompressedBlockStore {
    /// The compressed block of every (column, group), in memory or in the file
    source: BlockSource,
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
            source: BlockSource::Memory(all_blocks),
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
            source: BlockSource::Memory(blocks),
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

    /// A store whose blocks stay in the volume's file at `path`: `offsets`
    /// and `compressed_lens` locate every (column, group) block, read by
    /// position when a group is decoded, the file opened for the read.
    /// Nothing of the blocks is in RAM and no descriptor is held
    #[allow(clippy::too_many_arguments)]
    pub fn from_file(
        path: std::path::PathBuf,
        offsets: Vec<Vec<u64>>,
        compressed_lens: Vec<Vec<usize>>,
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
        Self {
            source: BlockSource::File {
                file: VolumeFile::shared(&path),
                offsets,
                lens: compressed_lens,
            },
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

    /// Whether the blocks live in the volume's file rather than in RAM
    pub fn is_file_backed(&self) -> bool {
        matches!(self.source, BlockSource::File { .. })
    }

    /// Removes the volume's file once its last holder lets go, whichever
    /// store holds it; false when the blocks are not in a file
    pub fn retire_file(&self) -> bool {
        match &self.source {
            BlockSource::File { file, .. } => {
                file.retire();
                true
            }
            BlockSource::Memory(_) => false,
        }
    }

    /// The number of groups of a column, from wherever the blocks live
    fn column_groups(&self, col_idx: usize) -> std::io::Result<usize> {
        let groups = match &self.source {
            BlockSource::Memory(blocks) => blocks.get(col_idx).map(Vec::len),
            BlockSource::File { lens, .. } => lens.get(col_idx).map(Vec::len),
        };
        groups.ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "column index out of range",
            )
        })
    }

    /// The compressed block of (column, group): a slice of the store's
    /// memory, or read from the file into `scratch`
    fn block<'s>(
        &'s self,
        col_idx: usize,
        group_idx: usize,
        scratch: &'s mut Vec<u8>,
    ) -> std::io::Result<&'s [u8]> {
        let missing =
            || std::io::Error::new(std::io::ErrorKind::InvalidInput, "group index out of range");
        match &self.source {
            BlockSource::Memory(blocks) => blocks
                .get(col_idx)
                .and_then(|col| col.get(group_idx))
                .map(Vec::as_slice)
                .ok_or_else(missing),
            BlockSource::File {
                file,
                offsets,
                lens,
            } => {
                let offset = *offsets
                    .get(col_idx)
                    .and_then(|col| col.get(group_idx))
                    .ok_or_else(missing)?;
                let len = *lens
                    .get(col_idx)
                    .and_then(|col| col.get(group_idx))
                    .ok_or_else(missing)?;
                scratch.clear();
                scratch
                    .try_reserve_exact(len)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::OutOfMemory, e))?;
                scratch.resize(len, 0);
                read_exact_at(&file.open()?, scratch, offset)?;
                Ok(scratch.as_slice())
            }
        }
    }

    /// Decompress a single column from RAM. Concatenates all row-group blocks.
    /// Runs at ~4 GB/s (LZ4 from RAM), typically <1ms per column.
    pub fn decompress_column(&self, col_idx: usize) -> std::io::Result<ColumnData> {
        let num_groups = self.column_groups(col_idx)?;
        let mut block_buf = Vec::new();
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
            let block = self.block(col_idx, 0, &mut block_buf)?;
            return self.decompress_block(col_idx, 0, block, type_tag, num_groups, dict, ext_type);
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
                for gi in 0..num_groups {
                    let block = self.block(col_idx, gi, &mut block_buf)?;
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
                for gi in 0..num_groups {
                    let block = self.block(col_idx, gi, &mut block_buf)?;
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
                for gi in 0..num_groups {
                    let block = self.block(col_idx, gi, &mut block_buf)?;
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
                for gi in 0..num_groups {
                    let block = self.block(col_idx, gi, &mut block_buf)?;
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
                for gi in 0..num_groups {
                    let block = self.block(col_idx, gi, &mut block_buf)?;
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
                for gi in 0..num_groups {
                    let block = self.block(col_idx, gi, &mut block_buf)?;
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
        let num_groups = self.column_groups(col_idx)?;
        let mut block_buf = Vec::new();
        let block = self.block(col_idx, group_idx, &mut block_buf)?;
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
        let num_groups = self.column_groups(col_idx)?;
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
        self.column_groups(col_idx).unwrap_or(0)
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
        self.col_type_tags.len()
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
        if let BlockSource::Memory(blocks) = &self.source {
            for col_blocks in blocks {
                for block in col_blocks {
                    size += block.len();
                }
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
    /// Empty for a file-backed store, whose blocks are in the volume's file
    pub fn raw_blocks(&self) -> &[Vec<Vec<u8>>] {
        match &self.source {
            BlockSource::Memory(blocks) => blocks,
            BlockSource::File { .. } => &[],
        }
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

    /// A column already decoded here, without decoding it: None while the
    /// column is still in its compressed form or was never loaded
    pub fn resident(&self, idx: usize) -> Option<&ColumnData> {
        match self.slots.get(idx)?.get() {
            Some(Ok(col)) => Some(col),
            _ => None,
        }
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
    /// Positions sorted by row id when the ids are not ascending, built on
    /// first use; None when a binary search over `row_ids` works
    pub row_order: std::sync::OnceLock<Option<Box<[u32]>>>,
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
        // row_order: one u32 per row when the ids do not ascend
        if let Some(Some(order)) = self.row_order.get() {
            size += order.len() * 4;
        }
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
    /// The producer orders the rows itself, so ids need not ascend
    any_row_order: bool,
    // Sort tracking
    last_values: Vec<Option<i64>>,
    sorted: Vec<bool>,
    // Row count
    row_count: usize,
    /// Bloom filters fed as the cells arrive, when the producer asked for
    /// them; otherwise built over the columns at `finish`
    bloom: Option<Vec<super::column::ColumnBloomFilter>>,
    /// The row groups flushed out of the accumulators by a streaming
    /// producer, with their zone maps
    row_groups: Vec<super::column::RowGroupMeta>,
    /// Rows flushed out of the accumulators so far
    flushed_rows: usize,
}

/// One output column's cells for a batch of rows in the column's storage
/// form: what `VolumeBuilder::append_typed` takes instead of rows
pub enum TypedCells<'a> {
    Int64 {
        values: &'a [i64],
        nulls: &'a [bool],
    },
    Float64 {
        values: &'a [f64],
        nulls: &'a [bool],
    },
    TimestampNanos {
        values: &'a [i64],
        nulls: &'a [bool],
    },
    Boolean {
        values: &'a [bool],
        nulls: &'a [bool],
    },
    /// Ids in the builder's dictionary for the column, from `intern_text`
    Dictionary { ids: &'a [u32], nulls: &'a [bool] },
    /// Extension payloads without their type tag, as `ColumnData::Bytes`
    /// holds them
    Bytes {
        data: &'a [u8],
        offsets: &'a [(u64, u64)],
        nulls: &'a [bool],
    },
}

impl<'a> TypedCells<'a> {
    pub fn len(&self) -> usize {
        match self {
            TypedCells::Int64 { nulls, .. }
            | TypedCells::Float64 { nulls, .. }
            | TypedCells::TimestampNanos { nulls, .. }
            | TypedCells::Boolean { nulls, .. }
            | TypedCells::Dictionary { nulls, .. }
            | TypedCells::Bytes { nulls, .. } => nulls.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The cells at `range`; a bytes column keeps its whole payload with
    /// the offsets of the range, which every consumer reads by offset
    pub fn slice(&self, range: std::ops::Range<usize>) -> TypedCells<'a> {
        match self {
            TypedCells::Int64 { values, nulls } => TypedCells::Int64 {
                values: &values[range.clone()],
                nulls: &nulls[range],
            },
            TypedCells::Float64 { values, nulls } => TypedCells::Float64 {
                values: &values[range.clone()],
                nulls: &nulls[range],
            },
            TypedCells::TimestampNanos { values, nulls } => TypedCells::TimestampNanos {
                values: &values[range.clone()],
                nulls: &nulls[range],
            },
            TypedCells::Boolean { values, nulls } => TypedCells::Boolean {
                values: &values[range.clone()],
                nulls: &nulls[range],
            },
            TypedCells::Dictionary { ids, nulls } => TypedCells::Dictionary {
                ids: &ids[range.clone()],
                nulls: &nulls[range],
            },
            TypedCells::Bytes {
                data,
                offsets,
                nulls,
            } => TypedCells::Bytes {
                data,
                offsets: &offsets[range.clone()],
                nulls: &nulls[range],
            },
        }
    }
}

/// Moves `min` and `max` to include `value`, the way `add_row` does cell
/// by cell: a null extent takes the value, otherwise the value replaces
/// the extent it lies beyond
fn extend_extents(min: &mut Value, max: &mut Value, value: &Value) {
    if min.is_null() {
        *min = value.clone();
        *max = value.clone();
        return;
    }
    if let Ok(std::cmp::Ordering::Less) = value.compare(min) {
        *min = value.clone();
    }
    if let Ok(std::cmp::Ordering::Greater) = value.compare(max) {
        *max = value.clone();
    }
}

/// Original and lowercase names to positions, one entry when they agree
fn column_name_map(column_names: &[String]) -> AHashMap<SmartString, usize> {
    column_names
        .iter()
        .enumerate()
        .flat_map(|(i, name)| {
            let lower = SmartString::from(name.to_lowercase());
            let original = SmartString::from(name.as_str());
            if lower == original {
                vec![(lower, i)]
            } else {
                vec![(original, i), (lower, i)]
            }
        })
        .collect()
}

/// The zone map of one row group over its typed cells, as
/// `ColumnData::zone_map_for_range` computes it over a column: text by the
/// dictionary's strings, an extension column without extents
fn group_zone_map(
    cells: &TypedCells<'_>,
    dict: Option<&[SmartString]>,
    ext_type: DataType,
) -> ZoneMap {
    let row_count = cells.len() as u32;
    let mut null_count = 0u32;
    let (min, max) = match cells {
        TypedCells::Int64 { values, nulls } => {
            let (mut lo, mut hi, mut seen) = (i64::MAX, i64::MIN, false);
            for (&v, &is_null) in values.iter().zip(*nulls) {
                if is_null {
                    null_count += 1;
                } else {
                    lo = lo.min(v);
                    hi = hi.max(v);
                    seen = true;
                }
            }
            if seen {
                (Value::Integer(lo), Value::Integer(hi))
            } else {
                (
                    Value::Null(DataType::Integer),
                    Value::Null(DataType::Integer),
                )
            }
        }
        TypedCells::Float64 { values, nulls } => {
            let (mut lo, mut hi, mut seen) = (f64::INFINITY, f64::NEG_INFINITY, false);
            for (&v, &is_null) in values.iter().zip(*nulls) {
                if is_null {
                    null_count += 1;
                } else if !v.is_nan() {
                    if !seen || v < lo {
                        lo = v;
                    }
                    if !seen || v > hi {
                        hi = v;
                    }
                    seen = true;
                }
            }
            if seen {
                (Value::Float(lo), Value::Float(hi))
            } else {
                (Value::Null(DataType::Float), Value::Null(DataType::Float))
            }
        }
        TypedCells::TimestampNanos { values, nulls } => {
            let (mut lo, mut hi, mut seen) = (i64::MAX, i64::MIN, false);
            for (&v, &is_null) in values.iter().zip(*nulls) {
                if is_null {
                    null_count += 1;
                } else {
                    lo = lo.min(v);
                    hi = hi.max(v);
                    seen = true;
                }
            }
            if seen {
                (timestamp_value(lo), timestamp_value(hi))
            } else {
                (
                    Value::Null(DataType::Timestamp),
                    Value::Null(DataType::Timestamp),
                )
            }
        }
        TypedCells::Boolean { values, nulls } => {
            let (mut has_true, mut has_false) = (false, false);
            for (&v, &is_null) in values.iter().zip(*nulls) {
                if is_null {
                    null_count += 1;
                } else if v {
                    has_true = true;
                } else {
                    has_false = true;
                }
            }
            match (has_false, has_true) {
                (true, true) => (Value::Boolean(false), Value::Boolean(true)),
                (true, false) => (Value::Boolean(false), Value::Boolean(false)),
                (false, true) => (Value::Boolean(true), Value::Boolean(true)),
                (false, false) => (
                    Value::Null(DataType::Boolean),
                    Value::Null(DataType::Boolean),
                ),
            }
        }
        TypedCells::Dictionary { ids, nulls } => {
            let dict = dict.unwrap_or(&[]);
            let (mut lo, mut hi): (Option<&str>, Option<&str>) = (None, None);
            for (&id, &is_null) in ids.iter().zip(*nulls) {
                if is_null {
                    null_count += 1;
                    continue;
                }
                let s = dict.get(id as usize).map(|s| s.as_str()).unwrap_or("");
                lo = Some(match lo {
                    Some(cur) if cur <= s => cur,
                    _ => s,
                });
                hi = Some(match hi {
                    Some(cur) if cur >= s => cur,
                    _ => s,
                });
            }
            match (lo, hi) {
                (Some(lo), Some(hi)) => (Value::text(lo), Value::text(hi)),
                _ => (Value::Null(DataType::Text), Value::Null(DataType::Text)),
            }
        }
        TypedCells::Bytes { nulls, .. } => {
            null_count = nulls.iter().filter(|n| **n).count() as u32;
            (Value::Null(ext_type), Value::Null(ext_type))
        }
    };
    ZoneMap {
        min,
        max,
        null_count,
        row_count,
    }
}

/// The timestamp value `nanos` decodes to, as `ColumnData::get_value` reads
/// it; null when the nanos fall outside the calendar
pub(crate) fn timestamp_value(nanos: i64) -> Value {
    let secs = nanos.div_euclid(1_000_000_000);
    let sub_nanos = nanos.rem_euclid(1_000_000_000) as u32;
    match chrono::TimeZone::timestamp_opt(&chrono::Utc, secs, sub_nanos) {
        chrono::LocalResult::Single(dt) => Value::Timestamp(dt),
        _ => Value::Null(DataType::Timestamp),
    }
}

/// Positions of `ids` sorted by id
pub(crate) fn row_order_of(ids: &[i64]) -> Box<[u32]> {
    let mut order: Vec<u32> = (0..ids.len() as u32).collect();
    order.sort_unstable_by_key(|&i| ids[i as usize]);
    order.into_boxed_slice()
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
            any_row_order: false,
            last_values,
            sorted,
            row_count: 0,
            bloom: None,
            row_groups: Vec::new(),
            flushed_rows: 0,
        }
    }

    /// Feed the bloom filters as cells arrive, from `add_row` and
    /// `append_typed` alike, instead of building them over the columns at
    /// the end; a streaming producer, whose columns leave the
    /// accumulators group by group, needs this
    pub fn feed_bloom_filters(&mut self, expected_rows: usize) {
        self.bloom = Some(
            (0..self.num_cols)
                .map(|_| super::column::ColumnBloomFilter::new(expected_rows.max(1)))
                .collect(),
        );
    }

    /// Rows added so far
    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// A text column's dictionary so far; None for another column
    pub fn dictionary(&self, col_idx: usize) -> Option<&[SmartString]> {
        match self.col_storage.get(col_idx) {
            Some(StorageKind::Dictionary(idx)) => Some(&self.dict_tables[*idx]),
            _ => None,
        }
    }

    /// Rows in the accumulators, not yet flushed
    pub fn group_len(&self) -> usize {
        self.row_count - self.flushed_rows
    }

    /// Hands the accumulated rows to `emit` column by column as typed
    /// cells, records them as one row group with its zone maps, and
    /// clears the accumulators, capacity kept. The row ids, dictionaries,
    /// extents, stats and sortedness stay: they are the volume's
    pub fn flush_group(
        &mut self,
        mut emit: impl FnMut(usize, TypedCells<'_>) -> Result<()>,
    ) -> Result<()> {
        let rows = self.group_len();
        if rows == 0 {
            return Ok(());
        }
        let mut zone_maps = Vec::with_capacity(self.num_cols);
        for col_idx in 0..self.num_cols {
            let cells = self.group_cells(col_idx);
            let dict = match self.col_storage[col_idx] {
                StorageKind::Dictionary(idx) => Some(self.dict_tables[idx].as_slice()),
                _ => None,
            };
            let ext_type = match self.col_storage[col_idx] {
                StorageKind::Bytes(_, ext_type) => ext_type,
                _ => DataType::Null,
            };
            zone_maps.push(group_zone_map(&cells, dict, ext_type));
            emit(col_idx, cells)?;
        }
        self.row_groups.push(super::column::RowGroupMeta {
            start_idx: self.flushed_rows as u32,
            end_idx: self.row_count as u32,
            zone_maps,
        });
        self.flushed_rows = self.row_count;
        for nulls in &mut self.null_cols {
            nulls.clear();
        }
        for v in &mut self.int_cols {
            v.clear();
        }
        for v in &mut self.float_cols {
            v.clear();
        }
        for v in &mut self.ts_cols {
            v.clear();
        }
        for v in &mut self.bool_cols {
            v.clear();
        }
        for v in &mut self.dict_cols {
            v.clear();
        }
        for (data, offsets) in &mut self.bytes_cols {
            data.clear();
            offsets.clear();
        }
        Ok(())
    }

    /// The accumulated cells of one column in storage form
    fn group_cells(&self, col_idx: usize) -> TypedCells<'_> {
        let nulls = &self.null_cols[col_idx];
        match self.col_storage[col_idx] {
            StorageKind::Int64(idx) => TypedCells::Int64 {
                values: &self.int_cols[idx],
                nulls,
            },
            StorageKind::Float64(idx) => TypedCells::Float64 {
                values: &self.float_cols[idx],
                nulls,
            },
            StorageKind::Timestamp(idx) => TypedCells::TimestampNanos {
                values: &self.ts_cols[idx],
                nulls,
            },
            StorageKind::Boolean(idx) => TypedCells::Boolean {
                values: &self.bool_cols[idx],
                nulls,
            },
            StorageKind::Dictionary(idx) => TypedCells::Dictionary {
                ids: &self.dict_cols[idx],
                nulls,
            },
            StorageKind::Bytes(idx, _) => TypedCells::Bytes {
                data: &self.bytes_cols[idx].0,
                offsets: &self.bytes_cols[idx].1,
                nulls,
            },
        }
    }

    /// The column's storage type tag and extension type tag, as the
    /// metadata records them
    pub fn column_kind(&self, col_idx: usize) -> (u8, u8) {
        match self.col_storage[col_idx] {
            StorageKind::Int64(_) => (super::format::COL_INT64, 0),
            StorageKind::Float64(_) => (super::format::COL_FLOAT64, 0),
            StorageKind::Timestamp(_) => (super::format::COL_TIMESTAMP, 0),
            StorageKind::Boolean(_) => (super::format::COL_BOOLEAN, 0),
            StorageKind::Dictionary(_) => (COL_DICTIONARY, 0),
            StorageKind::Bytes(_, ext_type) => (COL_BYTES, ext_type as u8),
        }
    }

    /// The volume's metadata and dictionaries once every row has been
    /// flushed by a streaming producer: what `finish` computes, without
    /// the columns. Row groups are kept only for a volume of more than
    /// one group, as `finish` keeps them
    pub fn finish_streamed(mut self) -> Result<(VolumeMeta, Vec<Vec<SmartString>>)> {
        if self.flushed_rows != self.row_count {
            return Err(Error::internal("rows left in the accumulators at finish"));
        }
        let bloom_filters = self
            .bloom
            .take()
            .ok_or_else(|| Error::internal("streamed volume without fed bloom filters"))?;
        let row_order = self.row_order()?;
        let column_names: Vec<String> =
            self.schema.columns.iter().map(|c| c.name.clone()).collect();
        let column_types: Vec<DataType> = self.schema.columns.iter().map(|c| c.data_type).collect();
        let column_name_map = column_name_map(&column_names);
        let row_groups = if self.row_count > super::column::ROW_GROUP_SIZE {
            std::mem::take(&mut self.row_groups)
        } else {
            Vec::new()
        };
        let meta = VolumeMeta {
            zone_maps: std::mem::take(&mut self.zone_maps),
            bloom_filters,
            stats: std::mem::replace(&mut self.stats, VolumeAggregateStats::new(0)),
            row_count: self.row_count,
            column_names,
            column_types,
            row_ids: std::mem::take(&mut self.row_ids),
            row_order: std::sync::OnceLock::from(row_order),
            sorted_columns: std::mem::take(&mut self.sorted),
            column_name_map,
            row_groups,
        };
        Ok((meta, std::mem::take(&mut self.dict_tables)))
    }

    /// The permutation lookups need when the row ids do not ascend; an
    /// error when they do not and the producer did not say so
    fn row_order(&self) -> Result<Option<Box<[u32]>>> {
        if self.row_ids.windows(2).all(|w| w[0] < w[1]) {
            return Ok(None);
        }
        if !self.any_row_order {
            return Err(Error::internal(
                "volume rows were not added in ascending row id order",
            ));
        }
        let order = row_order_of(&self.row_ids);
        if order
            .windows(2)
            .any(|w| self.row_ids[w[0] as usize] == self.row_ids[w[1] as usize])
        {
            return Err(Error::internal("volume rows repeat a row id"));
        }
        Ok(Some(order))
    }

    /// Accept rows in the order the producer chose; lookups by row id then
    /// go through a sorted permutation instead of the ids themselves
    pub fn allow_any_row_order(&mut self) {
        self.any_row_order = true;
    }

    /// The id of `text` in the column's dictionary, added when new. A new
    /// entry also moves the column's text extents, since the minimum and
    /// maximum of a text column change only with a new distinct value
    pub fn intern_text(&mut self, col_idx: usize, text: &str) -> Result<u32> {
        let idx = match self.col_storage.get(col_idx) {
            Some(StorageKind::Dictionary(idx)) => *idx,
            _ => return Err(Error::internal("intern_text on a column that is not text")),
        };
        if let Some(&id) = self.dict_maps[idx].get(text) {
            return Ok(id);
        }
        let id = self.dict_tables[idx].len() as u32;
        let entry = SmartString::from(text);
        self.dict_tables[idx].push(entry.clone());
        self.dict_maps[idx].insert(entry.clone(), id);
        let value = Value::Text(entry);
        let zone = &mut self.zone_maps[col_idx];
        extend_extents(&mut zone.min, &mut zone.max, &value);
        let stats = &mut self.stats.columns[col_idx];
        extend_extents(&mut stats.min, &mut stats.max, &value);
        Ok(id)
    }

    /// Appends a batch of rows given column by column in storage form, the
    /// volume `add_row` would build from the same rows: one entry per
    /// schema column, each with as many cells as `row_ids`, text as ids
    /// from `intern_text`
    pub fn append_typed(&mut self, row_ids: &[i64], columns: &[TypedCells<'_>]) -> Result<()> {
        let rows = row_ids.len();
        if columns.len() != self.num_cols {
            return Err(Error::internal(
                "typed batch does not cover the volume's columns",
            ));
        }
        for (col_idx, cells) in columns.iter().enumerate() {
            let matches = matches!(
                (self.col_storage[col_idx], cells),
                (StorageKind::Int64(_), TypedCells::Int64 { .. })
                    | (StorageKind::Float64(_), TypedCells::Float64 { .. })
                    | (StorageKind::Timestamp(_), TypedCells::TimestampNanos { .. })
                    | (StorageKind::Boolean(_), TypedCells::Boolean { .. })
                    | (StorageKind::Dictionary(_), TypedCells::Dictionary { .. })
                    | (StorageKind::Bytes(..), TypedCells::Bytes { .. })
            );
            if !matches || cells.len() != rows {
                return Err(Error::internal(
                    "typed batch column does not match the column's storage",
                ));
            }
        }
        for (col_idx, cells) in columns.iter().enumerate() {
            match (self.col_storage[col_idx], cells) {
                (StorageKind::Int64(idx), TypedCells::Int64 { values, nulls }) => {
                    self.append_i64_cells(col_idx, idx, false, values, nulls)
                }
                (StorageKind::Timestamp(idx), TypedCells::TimestampNanos { values, nulls }) => {
                    self.append_i64_cells(col_idx, idx, true, values, nulls)
                }
                (StorageKind::Float64(idx), TypedCells::Float64 { values, nulls }) => {
                    self.append_f64_cells(col_idx, idx, values, nulls)
                }
                (StorageKind::Boolean(idx), TypedCells::Boolean { values, nulls }) => {
                    self.append_bool_cells(col_idx, idx, values, nulls)
                }
                (StorageKind::Dictionary(idx), TypedCells::Dictionary { ids, nulls }) => {
                    self.append_dict_cells(col_idx, idx, ids, nulls)
                }
                (
                    StorageKind::Bytes(idx, ext_type),
                    TypedCells::Bytes {
                        data,
                        offsets,
                        nulls,
                    },
                ) => self.append_bytes_cells(col_idx, idx, ext_type, data, offsets, nulls),
                _ => unreachable!("typed batch columns were checked against the storage"),
            }
        }
        self.row_ids.extend_from_slice(row_ids);
        self.stats.total_rows += rows as u64;
        self.stats.live_rows += rows as u64;
        self.row_count += rows;
        Ok(())
    }

    fn append_i64_cells(
        &mut self,
        col_idx: usize,
        idx: usize,
        timestamps: bool,
        values: &[i64],
        nulls: &[bool],
    ) {
        let target = if timestamps {
            &mut self.ts_cols[idx]
        } else {
            &mut self.int_cols[idx]
        };
        let zone = &mut self.zone_maps[col_idx];
        let null_col = &mut self.null_cols[col_idx];
        zone.row_count += values.len() as u32;
        let (mut lo, mut hi, mut sum, mut count) = (i64::MAX, i64::MIN, 0i128, 0u64);
        for (&v, &is_null) in values.iter().zip(nulls) {
            null_col.push(is_null);
            if is_null {
                zone.null_count += 1;
                self.sorted[col_idx] = false;
                target.push(0);
                continue;
            }
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v as i128;
            count += 1;
            if let Some(bloom) = &mut self.bloom {
                if timestamps {
                    bloom[col_idx].add_timestamp_nanos(v);
                } else {
                    bloom[col_idx].add_i64(v);
                }
            }
            if self.sorted[col_idx] {
                if let Some(last) = self.last_values[col_idx] {
                    if v < last {
                        self.sorted[col_idx] = false;
                    }
                }
                self.last_values[col_idx] = Some(v);
            }
            target.push(v);
        }
        if count == 0 {
            return;
        }
        let stats = &mut self.stats.columns[col_idx];
        stats.non_null_count += count;
        let (lo_value, hi_value) = if timestamps {
            (timestamp_value(lo), timestamp_value(hi))
        } else {
            stats.sum_int += sum;
            stats.numeric_count += count;
            (Value::Integer(lo), Value::Integer(hi))
        };
        extend_extents(&mut zone.min, &mut zone.max, &lo_value);
        extend_extents(&mut zone.min, &mut zone.max, &hi_value);
        extend_extents(&mut stats.min, &mut stats.max, &lo_value);
        extend_extents(&mut stats.min, &mut stats.max, &hi_value);
    }

    fn append_f64_cells(&mut self, col_idx: usize, idx: usize, values: &[f64], nulls: &[bool]) {
        let target = &mut self.float_cols[idx];
        let zone = &mut self.zone_maps[col_idx];
        let stats = &mut self.stats.columns[col_idx];
        let null_col = &mut self.null_cols[col_idx];
        zone.row_count += values.len() as u32;
        let (mut lo, mut hi, mut count) = (f64::INFINITY, f64::NEG_INFINITY, 0u64);
        for (&v, &is_null) in values.iter().zip(nulls) {
            null_col.push(is_null);
            if is_null {
                zone.null_count += 1;
                self.sorted[col_idx] = false;
                target.push(0.0);
                continue;
            }
            target.push(v);
            if let Some(bloom) = &mut self.bloom {
                bloom[col_idx].add_f64(v);
            }
            // NaN counts for nothing, as in add_row: it neither sums nor
            // bounds the column
            if v.is_nan() {
                continue;
            }
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
            // The running sum takes each value in row order, as add_row
            // does: a batch summed on its own can overflow where the
            // running sum does not
            stats.sum_float += v;
            count += 1;
        }
        if count == 0 {
            return;
        }
        stats.non_null_count += count;
        stats.numeric_count += count;
        let (lo_value, hi_value) = (Value::Float(lo), Value::Float(hi));
        extend_extents(&mut zone.min, &mut zone.max, &lo_value);
        extend_extents(&mut zone.min, &mut zone.max, &hi_value);
        extend_extents(&mut stats.min, &mut stats.max, &lo_value);
        extend_extents(&mut stats.min, &mut stats.max, &hi_value);
    }

    fn append_bool_cells(&mut self, col_idx: usize, idx: usize, values: &[bool], nulls: &[bool]) {
        let target = &mut self.bool_cols[idx];
        let zone = &mut self.zone_maps[col_idx];
        let null_col = &mut self.null_cols[col_idx];
        zone.row_count += values.len() as u32;
        let (mut any_true, mut any_false, mut count) = (false, false, 0u64);
        for (&v, &is_null) in values.iter().zip(nulls) {
            null_col.push(is_null);
            if is_null {
                zone.null_count += 1;
                self.sorted[col_idx] = false;
                target.push(false);
                continue;
            }
            any_true |= v;
            any_false |= !v;
            count += 1;
            if let Some(bloom) = &mut self.bloom {
                bloom[col_idx].add_bool(v);
            }
            target.push(v);
        }
        if count == 0 {
            return;
        }
        let stats = &mut self.stats.columns[col_idx];
        stats.non_null_count += count;
        stats.numeric_count += count;
        stats.sum_int += values
            .iter()
            .zip(nulls)
            .filter(|(v, is_null)| **v && !**is_null)
            .count() as i128;
        let (lo_value, hi_value) = (Value::Boolean(!any_false), Value::Boolean(any_true));
        extend_extents(&mut zone.min, &mut zone.max, &lo_value);
        extend_extents(&mut zone.min, &mut zone.max, &hi_value);
        extend_extents(&mut stats.min, &mut stats.max, &lo_value);
        extend_extents(&mut stats.min, &mut stats.max, &hi_value);
    }

    fn append_dict_cells(&mut self, col_idx: usize, idx: usize, ids: &[u32], nulls: &[bool]) {
        let target = &mut self.dict_cols[idx];
        let zone = &mut self.zone_maps[col_idx];
        let null_col = &mut self.null_cols[col_idx];
        zone.row_count += ids.len() as u32;
        let mut count = 0u64;
        for (&id, &is_null) in ids.iter().zip(nulls) {
            null_col.push(is_null);
            if is_null {
                zone.null_count += 1;
                self.sorted[col_idx] = false;
                target.push(0);
                continue;
            }
            count += 1;
            if let Some(bloom) = &mut self.bloom {
                if let Some(text) = self.dict_tables[idx].get(id as usize) {
                    bloom[col_idx].add_str(text.as_str());
                }
            }
            target.push(id);
        }
        // The extents moved when the ids were interned
        self.stats.columns[col_idx].non_null_count += count;
    }

    fn append_bytes_cells(
        &mut self,
        col_idx: usize,
        idx: usize,
        ext_type: DataType,
        data: &[u8],
        offsets: &[(u64, u64)],
        nulls: &[bool],
    ) {
        let (target_data, target_offsets) = &mut self.bytes_cols[idx];
        let zone = &mut self.zone_maps[col_idx];
        let stats = &mut self.stats.columns[col_idx];
        let null_col = &mut self.null_cols[col_idx];
        zone.row_count += offsets.len() as u32;
        for (&(offset, length), &is_null) in offsets.iter().zip(nulls) {
            null_col.push(is_null);
            if is_null {
                zone.null_count += 1;
                self.sorted[col_idx] = false;
                target_offsets.push((0, 0));
                continue;
            }
            let payload = &data[offset as usize..(offset + length) as usize];
            target_offsets.push((target_data.len() as u64, length));
            target_data.extend_from_slice(payload);
            stats.non_null_count += 1;
            if let Some(bloom) = &mut self.bloom {
                bloom[col_idx].add_extension_noop();
            }
            // Extension values compare equal or not at all, so the first
            // non-null value is the column's minimum and maximum for good
            if zone.min.is_null() || stats.min.is_null() {
                let mut tagged = Vec::with_capacity(1 + payload.len());
                tagged.push(ext_type as u8);
                tagged.extend_from_slice(payload);
                let value = Value::Extension(crate::common::CompactArc::from(tagged));
                if zone.min.is_null() {
                    zone.min = value.clone();
                    zone.max = value.clone();
                }
                if stats.min.is_null() {
                    stats.min = value.clone();
                    stats.max = value;
                }
            }
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
                    if let Some(bloom) = &mut self.bloom {
                        bloom[col_idx].add_i64(v);
                    }
                    self.int_cols[idx].push(v);
                }
                StorageKind::Float64(idx) => {
                    let v = match value {
                        Value::Float(f) => *f,
                        _ => 0.0,
                    };
                    if let Some(bloom) = &mut self.bloom {
                        bloom[col_idx].add_f64(v);
                    }
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
                    if let Some(bloom) = &mut self.bloom {
                        bloom[col_idx].add_timestamp_nanos(nanos);
                    }
                    self.ts_cols[idx].push(nanos);
                }
                StorageKind::Boolean(idx) => {
                    let v = match value {
                        Value::Boolean(b) => *b,
                        _ => false,
                    };
                    if let Some(bloom) = &mut self.bloom {
                        bloom[col_idx].add_bool(v);
                    }
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
                    if let Some(bloom) = &mut self.bloom {
                        bloom[col_idx].add_str(self.dict_tables[idx][dict_id as usize].as_str());
                    }
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
                    if let Some(bloom) = &mut self.bloom {
                        bloom[col_idx].add_extension_noop();
                    }
                    self.bytes_cols[idx].1.push((offset, length));
                }
            }
        }
        self.row_count += 1;
    }

    /// Freeze the builder into a FrozenVolume. Rows must have been added
    /// in ascending row id order; any other order is an error
    pub fn finish(mut self) -> Result<FrozenVolume> {
        debug_assert_eq!(self.row_ids.len(), self.row_count);
        if self.flushed_rows != 0 {
            return Err(Error::internal(
                "a builder whose groups were flushed does not finish in memory",
            ));
        }
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
        let fed = self.bloom.take();
        let bloom_filters: Vec<super::column::ColumnBloomFilter> = columns
            .iter()
            .enumerate()
            .map(|(col_idx, col)| {
                if let Some(fed) = &fed {
                    return fed[col_idx].clone();
                }
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

        // Row ids name the payload at each position: lookups binary search
        // ascending ids, and otherwise a permutation only a producer that
        // ordered the rows itself gets to need
        let row_order = self.row_order()?;
        let column_name_map = column_name_map(&column_names);

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

        Ok(FrozenVolume {
            columns: LazyColumns::eager(columns, column_types.clone()),
            meta: Arc::new(VolumeMeta {
                zone_maps: self.zone_maps,
                bloom_filters,
                stats: self.stats,
                row_count: self.row_count,
                column_names,
                column_types,
                row_ids: self.row_ids,
                row_order: std::sync::OnceLock::from(row_order),
                sorted_columns,
                column_name_map,
                row_groups,
            }),
            unique_indices: Arc::new(parking_lot::RwLock::new(rustc_hash::FxHashMap::default())),
            last_access_epoch: std::sync::atomic::AtomicU64::new(
                GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
            ),
        })
    }
}

/// Reads rows of one volume by index with the groups they need held
/// across the reads: a loop over many rows of a volume decodes each
/// group once, not once per row and column, and lets the groups go when
/// the reader does
pub struct RowReader {
    volume: Arc<FrozenVolume>,
    /// By physical column: the group held and its first row
    pinned: Vec<Option<(Arc<ColumnData>, usize)>>,
}

impl RowReader {
    pub fn new(volume: Arc<FrozenVolume>) -> Self {
        let columns = volume.columns.len();
        Self {
            volume,
            pinned: vec![None; columns],
        }
    }

    pub fn volume(&self) -> &Arc<FrozenVolume> {
        &self.volume
    }

    fn cell(&mut self, col_idx: usize, row_idx: usize) -> std::io::Result<Value> {
        if let Some(column) = self.volume.columns.resident(col_idx) {
            return Ok(column.get_value(row_idx));
        }
        let start = row_idx / ROW_GROUP_SIZE * ROW_GROUP_SIZE;
        if let Some((column, held)) = &self.pinned[col_idx] {
            if *held == start {
                return Ok(column.get_value(row_idx - start));
            }
        }
        let store = self.volume.columns.compressed_store().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "column data is not loaded")
        })?;
        let column = store.group_column(col_idx, row_idx / ROW_GROUP_SIZE)?;
        let value = column.get_value(row_idx - start);
        self.pinned[col_idx] = Some((column, start));
        Ok(value)
    }

    /// The row at `idx` through `mapping`, every schema column
    pub fn row(&mut self, idx: usize, mapping: &ColumnMapping) -> std::io::Result<Row> {
        if mapping.is_identity {
            let mut values = Vec::with_capacity(self.volume.columns.len());
            for col_idx in 0..self.volume.columns.len() {
                values.push(self.cell(col_idx, idx)?);
            }
            return Ok(Row::from_values(values));
        }
        let mut values = Vec::with_capacity(mapping.sources.len());
        for source in &mapping.sources {
            values.push(match source {
                ColSource::Volume(col_idx) => self.cell(*col_idx, idx)?,
                ColSource::Default(value) => value.clone(),
            });
        }
        Ok(Row::from_values(values))
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
    /// The schema's column names, by position, so a column named in a
    /// filter resolves through `sources` rather than by name in the volume,
    /// which may still hold a dropped column of that name, or a renamed
    /// column under its old name. Empty when the names match by position
    pub names: Vec<crate::common::SmartString>,
    /// True when every schema column maps 1:1 to the same volume column
    /// in the same order. When true, callers can skip the mapping and
    /// use get_row()/get_row_projected() directly.
    pub is_identity: bool,
}

impl ColumnMapping {
    /// The volume column a filter's column name stands for under the
    /// schema this mapping was computed for: by name in the volume when
    /// the mapping is the identity, through `sources` otherwise, since the
    /// volume may still hold a dropped column of the same name. None for a
    /// column the volume does not hold
    pub fn volume_column(&self, volume: &FrozenVolume, name: &str) -> Option<usize> {
        if self.names.is_empty() {
            return volume.column_index(name);
        }
        let position = self
            .names
            .iter()
            .position(|n| n.as_str().eq_ignore_ascii_case(name))?;
        match self.sources.get(position)? {
            ColSource::Volume(v) => Some(*v),
            ColSource::Default(_) => None,
        }
    }
}

/// The drop-list name under which a rename's schema version is kept: a
/// byte no column name can hold, then the rename's ordinal in the rename
/// list. It never matches a column, and it is short whatever the names
pub fn rename_marker(ordinal: usize) -> String {
    format!("\0{ordinal}")
}

/// The drop-list name under which the log position the manifest's renames
/// and drops reach is kept
pub const DDL_LSN_MARKER: &str = "\0lsn";

/// A change of the schema since a volume was sealed
enum SchemaEvent<'a> {
    Renamed { old: &'a str, new: &'a str },
    Dropped(&'a str),
}

/// The schema changes made after `volume_schema_version`, newest first,
/// or None when a rename carries no version marker (recorded before the
/// markers existed), in which case the history has no order to walk
fn schema_events_since<'a>(
    volume_schema_version: u64,
    dropped_columns: &'a [(crate::common::SmartString, u64)],
    column_renames: &'a [(crate::common::SmartString, crate::common::SmartString)],
) -> Option<Vec<(u64, usize, SchemaEvent<'a>)>> {
    let mut events = Vec::new();
    for (ordinal, (old, new)) in column_renames.iter().enumerate() {
        let marker = rename_marker(ordinal);
        let (_, version) = dropped_columns
            .iter()
            .find(|(name, _)| name.as_str() == marker)?;
        if *version > volume_schema_version {
            events.push((
                *version,
                ordinal,
                SchemaEvent::Renamed {
                    old: old.as_str(),
                    new: new.as_str(),
                },
            ));
        }
    }
    for (order, (name, version)) in dropped_columns.iter().enumerate() {
        if !name.starts_with('\0') && *version > volume_schema_version {
            events.push((*version, order, SchemaEvent::Dropped(name.as_str())));
        }
    }
    events.sort_by_key(|event| std::cmp::Reverse((event.0, event.1)));
    Some(events)
}

/// The volume column a schema column stands for, or None when the
/// column's identity began after the seal (a drop ended the identity that
/// carried the name, or the name never reached the volume). The name is
/// walked back through the changes since the seal, newest first: a rename
/// into the name takes the name it had before, a drop of the name ends
/// the walk
fn source_before_changes<'a>(
    name_now: &'a str,
    events: &[(u64, usize, SchemaEvent<'a>)],
    volume: &FrozenVolume,
) -> Option<usize> {
    let mut name = name_now;
    for (_, _, event) in events {
        match event {
            SchemaEvent::Renamed { old, new } if new.eq_ignore_ascii_case(name) => name = old,
            SchemaEvent::Dropped(dropped) if dropped.eq_ignore_ascii_case(name) => return None,
            _ => {}
        }
    }
    volume.column_index(name)
}

/// The resolution used before renames carried a version: one rename
/// record back, the renamed column's old slot taking precedence over a
/// column that reuses the name, and a name dropped at or after the seal
/// resolving to nothing
fn source_without_order(
    name_now: &str,
    volume_schema_version: u64,
    dropped_columns: &[(crate::common::SmartString, u64)],
    column_renames: &[(crate::common::SmartString, crate::common::SmartString)],
    volume: &FrozenVolume,
) -> Option<usize> {
    let was_dropped = dropped_columns
        .iter()
        .any(|(d, drop_ver)| d.as_str() == name_now && volume_schema_version <= *drop_ver);
    if was_dropped {
        return None;
    }
    column_renames
        .iter()
        .find(|(_, new)| new.as_str() == name_now)
        .and_then(|(old, _)| volume.column_index(old.as_str()))
        .or_else(|| volume.column_index(name_now))
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
    let events = schema_events_since(volume_schema_version, dropped_columns, column_renames);

    for (pos, col) in schema.columns.iter().enumerate() {
        let vol_idx = match &events {
            Some(events) => source_before_changes(&col.name_lower, events, volume),
            None => source_without_order(
                &col.name_lower,
                volume_schema_version,
                dropped_columns,
                column_renames,
                volume,
            ),
        };

        if let Some(vol_idx) = vol_idx {
            let type_matches = vol_idx < volume.meta.column_types.len()
                && volume.meta.column_types[vol_idx] == col.data_type;
            let already_used = used_vol_indices.contains(&vol_idx);
            if type_matches && !already_used {
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

    // The names travel with the mapping when a filter's column name may
    // not resolve by name in the volume: the schema differs in shape, or a
    // column was renamed in place (two renames can swap names and leave
    // every column where it was)
    let names_match = is_identity
        && schema
            .columns
            .iter()
            .zip(volume.meta.column_names.iter())
            .all(|(col, name)| col.name.eq_ignore_ascii_case(name));
    let names = if names_match {
        Vec::new()
    } else {
        schema
            .columns
            .iter()
            .map(|col| crate::common::SmartString::from(col.name.as_str()))
            .collect()
    };
    ColumnMapping {
        sources,
        names,
        is_identity,
    }
}

impl FrozenVolume {
    /// Borrow the resident physical row IDs.
    #[inline]
    pub fn row_ids(&self) -> std::io::Result<&[i64]> {
        Ok(&self.meta.row_ids)
    }

    /// Positions sorted by row id when the ids do not ascend; None when
    /// the ids themselves are in order. Decided once per volume
    pub fn row_order(&self) -> Option<&[u32]> {
        let ids = &self.meta.row_ids;
        self.meta
            .row_order
            .get_or_init(|| (!ids.windows(2).all(|w| w[0] < w[1])).then(|| row_order_of(ids)))
            .as_deref()
    }

    /// The smallest and largest row id held, without a scan once the
    /// order is known
    pub fn id_bounds(&self) -> Option<(i64, i64)> {
        let ids = &self.meta.row_ids;
        match self.row_order() {
            None => Some((*ids.first()?, *ids.last()?)),
            Some(order) => Some((ids[*order.first()? as usize], ids[*order.last()? as usize])),
        }
    }

    /// Whether the rows are held in the order of `key`, compared cell by
    /// cell in place; an empty key orders nothing and is always satisfied
    pub fn in_key_order(&self, key: &[usize]) -> std::io::Result<bool> {
        let n = self.meta.row_count;
        let mut columns = Vec::with_capacity(key.len());
        for &c in key {
            columns.push(self.columns.get(c)?);
        }
        for i in 1..n {
            for col in &columns {
                match col.compare_cells(i - 1, col, i) {
                    std::cmp::Ordering::Less => break,
                    std::cmp::Ordering::Equal => continue,
                    std::cmp::Ordering::Greater => return Ok(false),
                }
            }
        }
        Ok(true)
    }

    /// Carry the order decided for an earlier form of this volume over to
    /// this one, so a reload does not decide it again
    pub fn inherit_row_order(&self, from: &FrozenVolume) {
        if let Some(order) = from.meta.row_order.get() {
            let _ = self.meta.row_order.set(order.clone());
        }
    }

    /// Position of `row_id` in this volume in whatever order its rows were
    /// added: a binary search over ascending ids, otherwise over a
    /// permutation built on first use
    pub fn locate(&self, row_id: i64) -> Option<usize> {
        let ids = &self.meta.row_ids;
        let (Some(&first), Some(&last)) = (ids.first(), ids.last()) else {
            return None;
        };
        match self.row_order() {
            None => {
                if row_id < first || row_id > last {
                    return None;
                }
                ids.binary_search(&row_id).ok()
            }
            Some(order) => {
                if row_id < ids[order[0] as usize] || row_id > ids[order[order.len() - 1] as usize]
                {
                    return None;
                }
                order
                    .binary_search_by(|&i| ids[i as usize].cmp(&row_id))
                    .ok()
                    .map(|k| order[k] as usize)
            }
        }
    }

    /// Get a row using a precomputed column mapping.
    /// Materializes all schema columns through the mapping.
    pub fn get_row_mapped(&self, idx: usize, mapping: &ColumnMapping) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(mapping.sources.len());
        for src in &mapping.sources {
            values.push(match src {
                ColSource::Volume(vol_idx) => self.cell(*vol_idx, idx)?,
                ColSource::Default(val) => val.clone(),
            });
        }
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
        for &ci in col_indices {
            values.push(match &mapping.sources[ci] {
                ColSource::Volume(vol_idx) => self.cell(*vol_idx, idx)?,
                ColSource::Default(val) => val.clone(),
            });
        }
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
                self.cell(ci, idx)?
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
        for (ci, src) in mapping.sources.iter().enumerate() {
            values.push(if ci < needed.len() && needed[ci] {
                match src {
                    ColSource::Volume(vol_idx) => self.cell(*vol_idx, idx)?,
                    ColSource::Default(val) => val.clone(),
                }
            } else {
                match src {
                    ColSource::Volume(vol_idx) => Value::Null(self.columns.data_type(*vol_idx)),
                    ColSource::Default(val) => Value::Null(val.data_type()),
                }
            });
        }
        Ok(Row::from_values(values))
    }

    /// Get a row as a Vec of Values (for executor compatibility).
    pub fn get_row(&self, idx: usize) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(self.columns.len());
        for ci in 0..self.columns.len() {
            values.push(self.cell(ci, idx)?);
        }
        Ok(Row::from_values(values))
    }

    /// Get specific columns of a row (projection pushdown).
    pub fn get_row_projected(&self, idx: usize, col_indices: &[usize]) -> std::io::Result<Row> {
        let mut values = Vec::with_capacity(col_indices.len());
        for &col in col_indices {
            values.push(self.cell(col, idx)?);
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
                let vol_val = self.cell(ci, row_idx as usize)?;
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

    /// Calls `f` with the column's data from the group holding `from` on,
    /// each call with the first row's index and the rows the call covers:
    /// the whole column at once when it is decoded, otherwise one group
    /// at a time through the decoded group cache. `f` returns whether to
    /// go on
    pub fn for_each_group_from<E: From<std::io::Error>>(
        &self,
        col_idx: usize,
        from: usize,
        mut f: impl FnMut(usize, &ColumnData) -> std::result::Result<bool, E>,
    ) -> std::result::Result<(), E> {
        if let Some(column) = self.columns.resident(col_idx) {
            f(0, column)?;
            return Ok(());
        }
        let store = self.columns.compressed_store().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "column data is not loaded")
        })?;
        let groups = store.num_groups(col_idx);
        for group in from / ROW_GROUP_SIZE..groups {
            let column = store.group_column(col_idx, group)?;
            if !f(group * ROW_GROUP_SIZE, &column)? {
                return Ok(());
            }
        }
        Ok(())
    }

    /// The first row index whose value is at least `target` in a sorted
    /// integer or timestamp column, None when every value is below it.
    /// The row groups' zone maps name the group that holds it, and only
    /// that group is decoded, through the decoded group cache
    pub fn first_index_ge(&self, col_idx: usize, target: i64) -> std::io::Result<Option<usize>> {
        if let Some(column) = self.columns.resident(col_idx) {
            let first = column.binary_search_ge(target);
            return Ok((first < column.len()).then_some(first));
        }
        let store = self.columns.compressed_store().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "column data is not loaded")
        })?;
        for group in 0..store.num_groups(col_idx) {
            let below_target = self
                .meta
                .row_groups
                .get(group)
                .and_then(|rg| rg.zone_maps.get(col_idx))
                .and_then(|zm| match &zm.max {
                    Value::Integer(max) => Some(*max),
                    Value::Timestamp(ts) => ts.timestamp_nanos_opt(),
                    _ => None,
                })
                .is_some_and(|max| target > max);
            if below_target {
                continue;
            }
            let column = store.group_column(col_idx, group)?;
            let local = column.binary_search_ge(target);
            if local < column.len() {
                return Ok(Some(group * ROW_GROUP_SIZE + local));
            }
        }
        Ok(None)
    }

    /// One cell, read from the column already decoded or from the row's
    /// group through the decoded group cache, never by decoding the
    /// column whole
    pub fn cell(&self, col_idx: usize, row_idx: usize) -> std::io::Result<Value> {
        if let Some(column) = self.columns.resident(col_idx) {
            return Ok(column.get_value(row_idx));
        }
        let store = self.columns.compressed_store().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "column data is not loaded")
        })?;
        let group = store.group_column(col_idx, row_idx / ROW_GROUP_SIZE)?;
        Ok(group.get_value(row_idx % ROW_GROUP_SIZE))
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

    /// Has the volume's file removed once its last holder lets go, so a
    /// reader still holding the volume, or a store reloaded from the
    /// same file, keeps reading it; false when the volume holds no
    /// file-backed store, in which case the caller removes the file itself
    pub fn retire_file(&self) -> bool {
        self.columns
            .compressed_store()
            .is_some_and(|store| store.retire_file())
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

    fn test_schema() -> Schema {
        SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("time", DataType::Timestamp, false, false)
            .column("exchange", DataType::Text, false, false)
            .column("price", DataType::Float, false, false)
            .build()
    }

    #[test]
    fn a_read_during_a_move_waits_for_it() {
        let dir = tempfile::tempdir().unwrap();
        let old_dir = dir.path().join("old");
        let new_dir = dir.path().join("new");
        std::fs::create_dir(&old_dir).unwrap();
        std::fs::write(old_dir.join("v.vol"), b"x").unwrap();
        let file = VolumeFile::shared(&old_dir.join("v.vol"));
        let reader = Arc::clone(&file);
        let (tx, rx) = std::sync::mpsc::channel();
        let read = std::sync::Mutex::new(None);
        VolumeFile::relocate(&old_dir, &new_dir, || {
            let reader = Arc::clone(&reader);
            let tx = tx.clone();
            *read.lock().unwrap() = Some(std::thread::spawn(move || {
                let opened = reader.open().map(|_| reader.path());
                tx.send(()).unwrap();
                opened
            }));
            // The read waits until the move is done
            assert!(rx
                .recv_timeout(std::time::Duration::from_millis(200))
                .is_err());
            std::fs::rename(&old_dir, &new_dir)
        })
        .unwrap();
        let opened = read.lock().unwrap().take().unwrap().join().unwrap();
        assert_eq!(opened.unwrap(), new_dir.join("v.vol"));
        assert_eq!(file.path(), new_dir.join("v.vol"));
    }

    #[test]
    fn a_holder_letting_go_during_a_move_does_not_block_it() {
        let dir = tempfile::tempdir().unwrap();
        let old_dir = dir.path().join("old");
        let new_dir = dir.path().join("new");
        std::fs::create_dir(&old_dir).unwrap();
        std::fs::write(old_dir.join("v.vol"), b"x").unwrap();
        let file = VolumeFile::shared(&old_dir.join("v.vol"));
        file.retire();
        let moved = std::thread::spawn(move || {
            VolumeFile::relocate(&old_dir, &new_dir, || {
                drop(file);
                std::fs::rename(&old_dir, &new_dir)
            })
            .unwrap();
            new_dir
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !moved.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(
            moved.is_finished(),
            "the move waits on its own registry lock"
        );
        let new_dir = moved.join().unwrap();
        // The retired file went with the last holder, at its new place
        assert!(!new_dir.join("v.vol").exists());
        assert!(!dir.path().join("old").join("v.vol").exists());
    }

    #[test]
    fn a_file_whose_last_holder_lets_go_during_a_move_is_removed_at_its_new_place() {
        let dir = tempfile::tempdir().unwrap();
        let old_dir = dir.path().join("old");
        let new_dir = dir.path().join("new");
        std::fs::create_dir(&old_dir).unwrap();
        std::fs::write(old_dir.join("v.vol"), b"x").unwrap();
        let file = VolumeFile::shared(&old_dir.join("v.vol"));
        file.retire();
        let gone = Arc::downgrade(&file);
        // The moves hold the registry; the last holder lets go meanwhile,
        // its drop waiting on the lock
        let mut registry = VOLUME_FILES.lock();
        let letting_go = std::thread::spawn(move || drop(file));
        while gone.strong_count() > 0 {
            std::thread::yield_now();
        }
        // There, back, and there again: the same names twice over
        for (from, to) in [
            (&old_dir, &new_dir),
            (&new_dir, &old_dir),
            (&old_dir, &new_dir),
        ] {
            let (result, held) =
                VolumeFile::relocate_in(&mut registry, from, to, || std::fs::rename(from, to));
            result.unwrap();
            assert!(held.is_empty());
        }
        drop(registry);
        letting_go.join().unwrap();
        assert!(!new_dir.join("v.vol").exists());
        assert!(!old_dir.exists());
        let registry = VOLUME_FILES.lock();
        assert!(registry.letting_go.is_empty());
        assert!(!registry.files.contains_key(&new_dir.join("v.vol")));
    }

    #[test]
    fn a_row_reader_decodes_each_group_once_for_a_loop() {
        use crate::storage::volume::group_cache::DECODED_GROUPS;
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build();
        let rows = ROW_GROUP_SIZE as i64 + 100;
        let mut builder = VolumeBuilder::new(&schema);
        for i in 0..rows {
            builder.add_row(
                i,
                &Row::from_values(vec![Value::Integer(i), Value::text(format!("n{}", i % 7))]),
            );
        }
        let mut volume = builder.finish().unwrap();
        let (_, store) = crate::storage::volume::io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        let warm = Arc::new(volume.to_warm().unwrap());
        // A cache too small for both groups of both columns, so a reader
        // that let its groups go would decode them again row after row
        DECODED_GROUPS.set_budget_bytes(0);
        DECODED_GROUPS.set_budget_bytes(1);
        let before = DECODED_GROUPS.stats().misses;
        let mut reader = RowReader::new(Arc::clone(&warm));
        for i in (0..rows as usize).step_by(97) {
            let row = reader
                .row(
                    i,
                    &ColumnMapping {
                        sources: Vec::new(),
                        names: Vec::new(),
                        is_identity: true,
                    },
                )
                .unwrap();
            assert_eq!(row[0], Value::Integer(i as i64));
        }
        let misses = DECODED_GROUPS.stats().misses - before;
        // Two groups, two columns: each decoded once
        assert_eq!(misses, 4, "groups decoded {misses} times");
    }

    #[test]
    fn fed_bloom_filters_take_rows_added_either_way() {
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, true, false)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        builder.feed_bloom_filters(4);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::text("one")]),
        );
        let ids = [2i64];
        let nulls = [false];
        let name = builder.intern_text(1, "two").unwrap();
        builder
            .append_typed(
                &ids,
                &[
                    TypedCells::Int64 {
                        values: &ids,
                        nulls: &nulls,
                    },
                    TypedCells::Dictionary {
                        ids: &[name],
                        nulls: &nulls,
                    },
                ],
            )
            .unwrap();
        let volume = builder.finish().unwrap();
        assert!(volume.meta.bloom_filters[0].might_contain(&Value::Integer(1)));
        assert!(volume.meta.bloom_filters[0].might_contain(&Value::Integer(2)));
        assert!(volume.meta.bloom_filters[1].might_contain(&Value::text("one")));
        assert!(volume.meta.bloom_filters[1].might_contain(&Value::text("two")));
    }

    #[test]
    fn a_float_sum_runs_in_row_order_across_typed_batches() {
        // The volume's sum is what add_row reaches row by row: minus 1e308
        // and zeros in one batch, two 1e308 in the next. Summing the
        // second batch on its own overflows before it joins the running sum
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("v", DataType::Float, false, false)
            .build();
        let mut first = vec![0.0; 4096];
        first[0] = -1e308;
        let second = vec![1e308, 1e308];
        let mut by_rows = VolumeBuilder::new(&schema);
        let mut ids = Vec::new();
        for (i, v) in first.iter().chain(&second).enumerate() {
            by_rows.add_row(
                i as i64,
                &Row::from_values(vec![Value::Integer(i as i64), Value::Float(*v)]),
            );
            ids.push(i as i64);
        }
        let mut typed = VolumeBuilder::new(&schema);
        let nulls = vec![false; 4096];
        for (batch_ids, values) in [(&ids[..4096], &first[..]), (&ids[4096..], &second[..])] {
            let ints: Vec<i64> = batch_ids.to_vec();
            typed
                .append_typed(
                    batch_ids,
                    &[
                        TypedCells::Int64 {
                            values: &ints,
                            nulls: &nulls[..batch_ids.len()],
                        },
                        TypedCells::Float64 {
                            values,
                            nulls: &nulls[..batch_ids.len()],
                        },
                    ],
                )
                .unwrap();
        }
        let (by_rows, typed) = (by_rows.finish().unwrap(), typed.finish().unwrap());
        let (want, got) = (
            by_rows.meta.stats.columns[1].sum_float,
            typed.meta.stats.columns[1].sum_float,
        );
        assert!(want.is_finite());
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "typed sum {got}, by rows {want}"
        );
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

        let volume = builder.finish().unwrap();

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

    /// One sealed volume (id, a, c) with a = 1 and c = 9, read under
    /// schemas that reached their names by renames and drops
    fn sealed_ac() -> FrozenVolume {
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("a", DataType::Integer, false, false)
            .column("c", DataType::Integer, false, false)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![
                Value::Integer(1),
                Value::Integer(1),
                Value::Integer(9),
            ]),
        );
        builder.finish().unwrap()
    }

    fn sm(s: &str) -> crate::common::SmartString {
        crate::common::SmartString::from(s)
    }

    #[test]
    fn a_history_with_versions_is_walked_in_order_and_one_without_is_read_as_before() {
        let volume = sealed_ac();
        // Sealed at version 1; then a renamed to b (2), b dropped (3), c
        // renamed to b (4): the schema's b is the old c
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("b", DataType::Integer, false, false)
            .build();
        let renames = vec![(sm("a"), sm("b")), (sm("c"), sm("b"))];
        let drops = vec![
            (sm(&rename_marker(0)), 2),
            (sm("b"), 3),
            (sm(&rename_marker(1)), 4),
        ];
        let mapping = compute_column_mapping_with_drops(&schema, &volume, &drops, 1, &renames);
        assert!(matches!(mapping.sources[1], ColSource::Volume(2)));
        // The same records without the rename markers have no order: the
        // one-step reading of before applies, and b, dropped at or after
        // the seal, resolves to its default
        let drops = vec![(sm("b"), 3)];
        let mapping = compute_column_mapping_with_drops(&schema, &volume, &drops, 1, &renames);
        assert!(matches!(mapping.sources[1], ColSource::Default(_)));
        // A rename recorded in the version the seal carries happened
        // before the seal and is not walked
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("a", DataType::Integer, false, false)
            .column("c", DataType::Integer, false, false)
            .build();
        let renames = vec![(sm("x"), sm("a"))];
        let drops = vec![(sm(&rename_marker(0)), 1)];
        let mapping = compute_column_mapping_with_drops(&schema, &volume, &drops, 1, &renames);
        assert!(mapping.is_identity && matches!(mapping.sources[1], ColSource::Volume(1)));
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

        let volume = builder.finish().unwrap();
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

        let volume = builder.finish().unwrap();
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

        let volume = builder.finish().unwrap();

        // Project only id and price (columns 0 and 3)
        let row = volume.get_row_projected(0, &[0, 3]).unwrap();
        assert_eq!(row.len(), 2);
        assert_eq!(row.get(0), Some(&Value::Integer(1)));
        assert_eq!(row.get(1), Some(&Value::Float(100.0)));
    }
}
