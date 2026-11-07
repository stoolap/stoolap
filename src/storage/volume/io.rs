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

//! File I/O for frozen volumes.
//!
//! Handles writing volumes to disk and reading them back.
//! Volumes are written atomically (write to .tmp, then rename) to prevent
//! corruption from crashes during writes.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::core::Result;

use super::column::ROW_GROUP_SIZE;
use super::format::{deserialize_volume_metadata, serialize_volume_metadata};
use super::writer::{CompressedBlockStore, FrozenVolume, LazyColumns};

/// Volume file extension
pub(crate) const VOLUME_EXT: &str = "vol";

/// Magic bytes for V4 per-column per-group compressed format.
pub(crate) const V4_MAGIC: [u8; 4] = *b"STV4";

/// V4 format version. Bump when the metadata or block layout changes.
pub(crate) const V4_VERSION: u32 = 1;

/// Orders `file`'s writes before every write that follows, without
/// waiting for the disk to make them durable: a publication's steps
/// take this in turn, and the full sync of its last step, the manifest
/// directory's, makes them all durable at once. On macOS this is an
/// `F_BARRIERFSYNC` (a full sync when the file system has no barrier),
/// where a full sync flushes the drive's cache; elsewhere `sync_data`
pub(crate) fn sync_ordered(file: &std::fs::File) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_BARRIERFSYNC) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if !barrier_unsupported(error.raw_os_error()) {
            return Err(error);
        }
        file.sync_all()
    }
    #[cfg(not(target_os = "macos"))]
    {
        file.sync_data()
    }
}

/// Whether a failed barrier means the file system has none, so that a
/// full sync stands in: the kernel answers EINVAL or ENOTSUP, and an
/// older kernel hands the request to a file system that does not know
/// it, which answers ENOTTY; any other error is the write's own
#[cfg(any(target_os = "macos", all(test, unix)))]
fn barrier_unsupported(code: Option<i32>) -> bool {
    matches!(
        code,
        Some(libc::EINVAL) | Some(libc::ENOTSUP) | Some(libc::ENOTTY)
    )
}

/// Volume catalog filename
const CATALOG_FILE: &str = "volumes.catalog";

/// Write a frozen volume to disk atomically (V5 format, LZ4 compressed).
pub fn write_volume_to_disk(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
) -> Result<PathBuf> {
    let (path, _store) = write_volume_to_disk_opts(dir, table_name, volume_id, volume, true)?;
    Ok(path)
}

/// Write a frozen volume to disk atomically as V5, with optional LZ4
/// compression of its blocks (a block that does not shrink is stored raw).
/// Returns (path, CompressedBlockStore).
pub fn write_volume_to_disk_opts(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
    compress: bool,
) -> Result<(PathBuf, CompressedBlockStore)> {
    let (data, store) = serialize_v5_opts(volume, compress, volume_id)
        .map_err(|e| crate::core::Error::internal(format!("V5 serialize failed: {}", e)))?;
    let path = write_volume_file(dir, table_name, volume_id, data)?;
    Ok((path, store))
}

/// Writes a volume as V4, the format files written before V5 have: the
/// fixtures of tests that read or damage V4 files
#[doc(hidden)]
pub fn write_volume_to_disk_v4(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
) -> Result<PathBuf> {
    let (data, _) = serialize_v4_opts(volume, true)
        .map_err(|e| crate::core::Error::internal(format!("V4 serialize failed: {}", e)))?;
    write_volume_file(dir, table_name, volume_id, data)
}

/// Writes `data` as `dir/table_name/vol_<id>.vol` through a synced
/// temporary file, then syncs the directory
fn write_volume_file(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    data: Vec<u8>,
) -> Result<PathBuf> {
    let table_dir = dir.join(table_name);
    std::fs::create_dir_all(&table_dir)
        .map_err(|e| crate::core::Error::internal(format!("failed to create volume dir: {}", e)))?;

    let filename = format!("vol_{:016x}.{}", volume_id, VOLUME_EXT);
    let final_path = table_dir.join(&filename);
    let tmp_path = table_dir.join(format!("{}.tmp", filename));

    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
            crate::core::Error::internal(format!("failed to create volume tmp file: {}", e))
        })?;
        f.write_all(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to write volume file: {}", e))
        })?;
        sync_ordered(&f).map_err(|e| {
            crate::core::Error::internal(format!("failed to fsync volume tmp file: {}", e))
        })?;
    }
    drop(data);

    std::fs::rename(&tmp_path, &final_path).map_err(|e| {
        crate::core::Error::internal(format!("failed to rename volume file: {}", e))
    })?;

    #[cfg(not(windows))]
    if let Ok(d) = std::fs::File::open(&table_dir) {
        sync_ordered(&d).map_err(|e| {
            crate::core::Error::internal(format!("failed to fsync volume directory: {}", e))
        })?;
    }

    Ok(final_path)
}

/// Serializes a volume as V5: the header, the blocks group by group, then
/// the sections and the footer
pub(crate) fn serialize_v5_opts(
    vol: &FrozenVolume,
    compress: bool,
    volume_id: u64,
) -> std::io::Result<(Vec<u8>, CompressedBlockStore)> {
    let col_count = vol.columns.len();
    let group_size = vol.meta.group_rows;
    let num_groups = vol.meta.row_count.div_ceil(group_size);
    let meta = super::format::v5_volume_sections(vol)?;
    let meta_bytes: usize = meta.iter().map(|s| s.bytes.len()).sum();
    let store = CompressedBlockStore::compress_columns_opts(
        &vol.columns,
        vol.meta.row_count,
        group_size,
        compress,
    )?;
    let blocks = store.raw_blocks();
    let decoded = store.decompressed_lens();
    let block_bytes: usize = blocks.iter().flatten().map(Vec::len).sum();
    let mut buf = Vec::with_capacity(block_bytes + meta_bytes + 4096);
    buf.extend_from_slice(&super::v5::header());
    let mut entries = Vec::with_capacity(col_count * num_groups);
    for group in 0..num_groups {
        for (col, col_blocks) in blocks.iter().enumerate() {
            let block = &col_blocks[group];
            entries.push(super::v5::BlockEntry::new(
                buf.len() as u64,
                block,
                decoded[col][group],
                col,
                group,
            )?);
            buf.extend_from_slice(block);
        }
    }
    let pos = buf.len() as u64;
    let root = super::v5::Root {
        cols: col_count as u32,
        groups: num_groups as u32,
        group_rows: group_size as u32,
        rows: vol.meta.row_count as u64,
        volume_id,
        nonce: super::v5::new_nonce(),
    };
    let identity = super::v5::write_tail(&mut buf, pos, &root, meta, &entries)?;
    if vol.meta.file_identity.get_or_init(|| identity) != &identity {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "the volume already belongs to another file",
        ));
    }
    Ok((buf, store))
}

/// Serialize a FrozenVolume to V4 format.
///
/// Layout:
/// ```text
/// [STV4 (4)] [version (4)] [col_count (4)] [num_groups (4)] [meta_compressed_len (4)]
/// [LZ4(metadata)]
/// [block_index: (compressed_len: u64, decompressed_len: u64) * col_count * num_groups]
/// [LZ4 blocks: col_0_grp_0, col_0_grp_1, ..., col_N_grp_G]
/// [CRC32 (4)]
/// ```
/// Serialize a volume to V4 format bytes with LZ4 compression.
pub fn serialize_v4_public(vol: &FrozenVolume) -> std::io::Result<(Vec<u8>, CompressedBlockStore)> {
    serialize_v4_opts(vol, true)
}

/// Returns (file_bytes, CompressedBlockStore). The store can be used to register
/// a lazy volume without re-reading from disk.
fn serialize_v4_opts(
    vol: &FrozenVolume,
    compress: bool,
) -> std::io::Result<(Vec<u8>, CompressedBlockStore)> {
    use std::io::Write;

    let col_count = vol.columns.len();
    let group_size = vol.meta.group_rows;
    let num_groups = if vol.meta.row_count == 0 {
        0
    } else {
        vol.meta.row_count.div_ceil(group_size)
    };

    // 1. Serialize + compress metadata
    let meta_raw = serialize_volume_metadata(vol)?;
    let meta_compressed = lz4_flex::compress_prepend_size(&meta_raw);
    drop(meta_raw);

    // 2. Build CompressedBlockStore (compresses all column blocks when enabled)
    let store = CompressedBlockStore::compress_columns_opts(
        &vol.columns,
        vol.meta.row_count,
        group_size,
        compress,
    )?;

    // 3. Compute total size for pre-allocation
    let all_blocks = store.raw_blocks();
    let all_decomp_lens = store.decompressed_lens();
    let block_lens_size = col_count * num_groups * 16;
    let total_block_bytes: usize = all_blocks
        .iter()
        .flat_map(|c| c.iter())
        .map(|b| b.len())
        .sum();
    let total_size = 20 + meta_compressed.len() + block_lens_size + total_block_bytes + 4;
    let mut buf = Vec::with_capacity(total_size);

    // 4. Fixed header (20 bytes)
    buf.write_all(&V4_MAGIC)?;
    buf.write_all(&V4_VERSION.to_le_bytes())?;
    buf.write_all(&(col_count as u32).to_le_bytes())?;
    buf.write_all(&(num_groups as u32).to_le_bytes())?;
    buf.write_all(&(meta_compressed.len() as u32).to_le_bytes())?;

    // 5. Compressed metadata
    buf.write_all(&meta_compressed)?;
    drop(meta_compressed);

    // 6. Block index: (compressed_len: u64, decompressed_len: u64) pairs
    for (col_blocks, col_decomp) in all_blocks.iter().zip(all_decomp_lens.iter()) {
        for (block, &decomp_len) in col_blocks.iter().zip(col_decomp.iter()) {
            buf.write_all(&(block.len() as u64).to_le_bytes())?;
            buf.write_all(&(decomp_len as u64).to_le_bytes())?;
        }
    }

    // 7. Block data
    for col_blocks in all_blocks {
        for block in col_blocks {
            buf.write_all(block)?;
        }
    }

    // 8. Trailing CRC32
    let crc = crc32fast::hash(&buf);
    buf.write_all(&crc.to_le_bytes())?;

    Ok((buf, store))
}

/// The volume's rows per group: every group `size` rows from `i * size`
/// but the last, `size` a power of two up to `ROW_GROUP_SIZE`; None when
/// the groups do not tile the rows that way
fn group_geometry(row_count: usize, row_groups: &[super::column::RowGroupMeta]) -> Option<usize> {
    let Some(first) = row_groups.first() else {
        return (row_count <= ROW_GROUP_SIZE).then_some(ROW_GROUP_SIZE);
    };
    let size = first.end_idx.checked_sub(first.start_idx)? as usize;
    if !size.is_power_of_two() || size > ROW_GROUP_SIZE {
        return None;
    }
    let tiles = row_groups.len() == row_count.div_ceil(size)
        && row_groups.iter().enumerate().all(|(i, g)| {
            g.start_idx as usize == i * size
                && g.end_idx as usize == ((i + 1) * size).min(row_count)
        });
    tiles.then_some(size)
}

/// Verify V4 magic and CRC through the opened file.
/// The deferred block store retains `handle`; `path` is only for errors.
fn read_volume_v4(
    file: std::fs::File,
    handle: std::sync::Arc<super::writer::VolumeFile>,
    path: &Path,
) -> Result<FrozenVolume> {
    use std::io::Read;

    let inv = |msg: &str| crate::core::Error::internal(format!("V4: {}", msg));

    let file_len = usize::try_from(
        file.metadata()
            .map_err(|e| crate::core::Error::internal(format!("V4 stat {:?}: {}", path, e)))?
            .len(),
    )
    .map_err(|_| inv("file length exceeds address space"))?;
    if file_len < 24 {
        return Err(inv("file too small"));
    }

    let mut reader = std::io::BufReader::new(file);
    let mut hasher = crc32fast::Hasher::new();

    // Helper: read exact bytes and feed to CRC
    macro_rules! crc_read {
        ($buf:expr) => {{
            reader
                .read_exact($buf)
                .map_err(|e| crate::core::Error::internal(format!("V4 read: {}", e)))?;
            hasher.update($buf);
        }};
    }

    // 1. Fixed header (20 bytes)
    let mut header = [0u8; 20];
    crc_read!(&mut header);

    if header[0..4] != V4_MAGIC {
        return Err(inv("bad magic"));
    }
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    if version != V4_VERSION {
        return Err(inv(&format!("unsupported version {}", version)));
    }
    let col_count = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    let num_groups = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let meta_len = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
    let mut remaining = file_len - 24;
    if meta_len > remaining {
        return Err(inv("metadata length exceeds file payload"));
    }
    remaining -= meta_len;

    // 2. Compressed metadata (read into temp buffer, decompress, drop)
    let mut meta_compressed = vec![0u8; meta_len];
    crc_read!(&mut meta_compressed);

    // Parse prepended uncompressed size (4 bytes LE), then decompress_into
    // to avoid lz4_flex::decompress_size_prepended allocating a fresh Vec.
    let meta_raw = if meta_compressed.len() >= 4 {
        let uncomp_size = u32::from_le_bytes(meta_compressed[..4].try_into().unwrap()) as usize;
        // Each LZ4 length-extension byte adds at most 255 output bytes.
        if uncomp_size > (meta_compressed.len() - 4).saturating_mul(255)
            || uncomp_size > isize::MAX as usize
        {
            return Err(inv("metadata LZ4 size exceeds possible expansion"));
        }
        let mut buf = vec![0u8; uncomp_size];
        let decoded = lz4_flex::decompress_into(&meta_compressed[4..], &mut buf)
            .map_err(|e| inv(&format!("metadata LZ4: {}", e)))?;
        if decoded != uncomp_size {
            return Err(inv("metadata LZ4 decoded length mismatch"));
        }
        drop(meta_compressed);
        buf
    } else {
        drop(meta_compressed);
        return Err(inv("metadata too short for LZ4 size prefix"));
    };
    let meta = deserialize_volume_metadata(&meta_raw)
        .map_err(|e| crate::core::Error::internal(format!("V4 metadata: {}", e)))?;
    drop(meta_raw);

    if meta.col_type_tags.len() != col_count {
        return Err(inv(&format!(
            "col_count mismatch: header={}, metadata={}",
            col_count,
            meta.col_type_tags.len()
        )));
    }
    let group_size = group_geometry(meta.row_count, &meta.row_groups)
        .ok_or_else(|| inv("row groups do not tile the rows at one power-of-two size"))?;
    if num_groups != meta.row_count.div_ceil(group_size) {
        return Err(inv("block group count does not match row count"));
    }

    // 3. Block index: (compressed_len: u64, decompressed_len: u64) pairs
    let total_blocks = col_count
        .checked_mul(num_groups)
        .ok_or_else(|| inv("block count overflow"))?;
    let index_len = total_blocks
        .checked_mul(16)
        .filter(|&len| len <= remaining)
        .ok_or_else(|| inv("block index exceeds file payload"))?;
    remaining -= index_len;
    let mut index_buf = vec![0u8; index_len];
    crc_read!(&mut index_buf);

    let mut compressed_lens = Vec::with_capacity(total_blocks);
    let mut decompressed_lens_flat = Vec::with_capacity(total_blocks);
    for i in 0..total_blocks {
        let off = i * 16;
        let compressed_len = usize::try_from(u64::from_le_bytes(
            index_buf[off..off + 8].try_into().unwrap(),
        ))
        .map_err(|_| inv("compressed block length exceeds address space"))?;
        let decompressed_len = usize::try_from(u64::from_le_bytes(
            index_buf[off + 8..off + 16].try_into().unwrap(),
        ))
        .map_err(|_| inv("decoded block length exceeds address space"))?;
        if compressed_len > remaining {
            return Err(inv("compressed block exceeds file payload"));
        }
        remaining -= compressed_len;
        compressed_lens.push(compressed_len);
        decompressed_lens_flat.push(decompressed_len);
    }
    if remaining != 0 {
        return Err(inv("trailing bytes after volume payload"));
    }
    drop(index_buf);

    // 5. The blocks stay in the file: their offsets follow from the index
    //    (column-major, as written), and the rest of the file passes
    //    through the hasher in a small buffer so the whole-file CRC is
    //    verified without holding a block. A group is read by position
    //    when it is decoded
    let blocks_start = 20 + meta_len + index_len;
    let mut all_offsets: Vec<Vec<u64>> = Vec::with_capacity(col_count);
    let mut all_comp_lens: Vec<Vec<usize>> = Vec::with_capacity(col_count);
    let mut all_decomp_lens: Vec<Vec<usize>> = Vec::with_capacity(col_count);
    let mut block_idx = 0usize;
    let mut position = blocks_start as u64;
    for _col_idx in 0..col_count {
        let mut col_offsets = Vec::with_capacity(num_groups);
        let mut col_comp = Vec::with_capacity(num_groups);
        let mut col_lens = Vec::with_capacity(num_groups);
        for _gi in 0..num_groups {
            col_offsets.push(position);
            col_comp.push(compressed_lens[block_idx]);
            col_lens.push(decompressed_lens_flat[block_idx]);
            position += compressed_lens[block_idx] as u64;
            block_idx += 1;
        }
        all_offsets.push(col_offsets);
        all_comp_lens.push(col_comp);
        all_decomp_lens.push(col_lens);
    }
    let mut left = file_len - 24 - meta_len - index_len;
    let mut chunk = vec![0u8; (1 << 20).min(left.max(1))];
    while left > 0 {
        let take = chunk.len().min(left);
        crc_read!(&mut chunk[..take]);
        left -= take;
    }
    drop(chunk);

    // 6. Verify CRC32 (computed incrementally over everything we read)
    let mut crc_buf = [0u8; 4];
    reader
        .read_exact(&mut crc_buf)
        .map_err(|e| crate::core::Error::internal(format!("V4 CRC read: {}", e)))?;
    let stored_crc = u32::from_le_bytes(crc_buf);
    if hasher.finalize() != stored_crc {
        return Err(inv("CRC mismatch"));
    }
    drop(reader);

    // 7. Build the file-backed CompressedBlockStore + deferred LazyColumns.
    //    Columns start in the file. First scan decodes per group on
    //    demand. After all columns are accessed, is_eager flips to true
    //    (automatic hot promotion).
    Ok(file_volume(
        handle,
        meta,
        (all_offsets, all_comp_lens, all_decomp_lens),
        group_size,
        None,
    ))
}

/// A volume whose columns are read from `handle` a group at a time, each
/// block where `blocks` (offsets, stored and decoded lengths, by column
/// and group) places it, checked against `crcs` on every read when given
fn file_volume(
    handle: std::sync::Arc<super::writer::VolumeFile>,
    meta: super::format::VolumeMetadata,
    (all_offsets, all_comp_lens, all_decomp_lens): BlockPlaces,
    group_size: usize,
    crcs: Option<Vec<Vec<u32>>>,
) -> FrozenVolume {
    let col_data_types = meta.column_types.clone();
    let dict_ranges: Vec<(usize, usize, usize)> = {
        let mut ranges = Vec::new();
        let mut offset = 0usize;
        for (i, &count) in meta.col_dict_counts.iter().enumerate() {
            let count = count as usize;
            if count > 0 {
                ranges.push((i, offset, offset + count));
                offset += count;
            }
        }
        ranges
    };
    let mut store = CompressedBlockStore::from_shared_file(
        handle,
        all_offsets,
        all_comp_lens,
        all_decomp_lens,
        meta.col_type_tags.clone(),
        meta.col_ext_types.clone(),
        meta.shared_dict,
        dict_ranges,
        group_size,
        meta.row_count,
    );
    if let Some(crcs) = crcs {
        store.check_blocks(crcs);
    }
    let columns = LazyColumns::deferred(store, col_data_types);

    FrozenVolume {
        columns,
        meta: Arc::new(super::writer::VolumeMeta {
            zone_maps: meta.zone_maps,
            bloom_filters: meta.bloom_filters,
            stats: meta.stats,
            row_count: meta.row_count,
            column_names: meta.column_names,
            column_types: meta.column_types,
            row_ids: meta.row_ids,
            row_order: std::sync::OnceLock::new(),
            sorted_columns: meta.col_sorted,
            column_name_map: meta.column_name_map,
            row_groups: meta.row_groups,
            group_rows: group_size,
            file_identity: std::sync::OnceLock::new(),
        }),
        unique_indices: std::sync::Arc::new(super::writer::UniqueIndexes::new(
            rustc_hash::FxHashMap::default(),
        )),
        last_access_epoch: std::sync::atomic::AtomicU64::new(
            super::writer::GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed),
        ),
    }
}

/// Every block's offset, stored length and decoded length, by column and
/// group
type BlockPlaces = (Vec<Vec<u64>>, Vec<Vec<usize>>, Vec<Vec<usize>>);

/// Reads a V5 volume: the footer, the section table and the sections; the
/// blocks stay in the file, each checked against its CRC when read
fn read_volume_v5(
    file: std::fs::File,
    handle: std::sync::Arc<super::writer::VolumeFile>,
    path: &Path,
) -> Result<FrozenVolume> {
    use super::format::{
        V5_BLOOMS, V5_COLUMNS, V5_DICT, V5_GROUPS, V5_ROW_IDS, V5_STATS, V5_ZONES,
    };
    let fail = |e: std::io::Error| crate::core::Error::internal(format!("{:?}: {}", path, e));
    let inv = |msg: &str| crate::core::Error::internal(format!("{:?}: V5: {}", path, msg));
    let len = file.metadata().map_err(fail)?.len();
    let read_at = |buf: &mut [u8], at: u64| super::writer::read_exact_at(&file, buf, at);
    let known = [
        V5_COLUMNS, V5_GROUPS, V5_DICT, V5_ROW_IDS, V5_ZONES, V5_STATS,
    ];
    let mut opened = super::v5::open(len, &known, &[V5_BLOOMS], &read_at).map_err(fail)?;
    let super::format::V5Metadata { mut meta, bounds } =
        super::format::v5_metadata_from_sections(&mut opened.meta, &opened.root).map_err(fail)?;
    // A column's bloom words are read on its first probe
    let cols = meta.col_type_tags.len();
    let places = match opened.deferred.iter().find(|s| s.0 == V5_BLOOMS) {
        Some(&(_, enc, at, len)) => {
            super::format::v5_bloom_places(enc, at, len, cols, &read_at).map_err(fail)?
        }
        None => (0..cols).map(|_| None).collect(),
    };
    drop(file);
    meta.bloom_filters = super::writer::VolumeBlooms::in_file(&handle, places);
    let root = &opened.root;
    // One group spans every row; more tile them at one power-of-two size,
    // which their zone maps' bounds carry
    let group_size = match bounds.as_slice() {
        [] if meta.row_count == 0 => ROW_GROUP_SIZE,
        [(0, end)] if *end as usize == meta.row_count && meta.row_count <= ROW_GROUP_SIZE => {
            ROW_GROUP_SIZE
        }
        [_, _, ..] => group_geometry(meta.row_count, &meta.row_groups)
            .filter(|&size| size == root.group_rows as usize)
            .ok_or_else(|| inv("groups do not tile the rows at the root's group size"))?,
        _ => return Err(inv("group directory does not cover the rows")),
    };
    let places = opened
        .places(meta.col_type_tags.len(), bounds.len())
        .map_err(fail)?;
    let volume = file_volume(
        handle,
        meta,
        (places.offsets, places.stored, places.decoded),
        group_size,
        Some(places.crcs),
    );
    let _ = volume.meta.file_identity.set(opened.identity);
    Ok(volume)
}

/// Rewrites the volume file at `path` with the same rows in row groups of
/// `group_rows`: the files tests read when a volume was written with a
/// group size other than the writer's
#[cfg(feature = "test-failpoints")]
pub fn rewrite_volume_with_group_rows(path: &Path, group_rows: usize) -> Result<()> {
    let volume = read_volume_from_disk(path)?;
    let columns = (0..volume.columns.len())
        .map(|i| volume.columns.get(i).cloned())
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| crate::core::Error::internal(format!("rewrite {:?}: {}", path, e)))?;
    let mut meta = (*volume.meta).clone();
    meta.file_identity = std::sync::OnceLock::new();
    meta.row_groups = super::writer::row_groups_of(&columns, meta.row_count, group_rows);
    meta.group_rows = group_rows;
    let rewritten = FrozenVolume {
        columns: LazyColumns::eager(columns, meta.column_types.clone()),
        meta: Arc::new(meta),
        unique_indices: Arc::new(super::writer::UniqueIndexes::new(
            rustc_hash::FxHashMap::default(),
        )),
        last_access_epoch: std::sync::atomic::AtomicU64::new(0),
    };
    drop(volume);
    let (data, _) = serialize_v4_opts(&rewritten, true)
        .map_err(|e| crate::core::Error::internal(format!("rewrite {:?}: {}", path, e)))?;
    let tmp = path.with_extension("vol.tmp");
    std::fs::write(&tmp, &data)
        .and_then(|()| std::fs::rename(&tmp, path))
        .map_err(|e| crate::core::Error::internal(format!("rewrite {:?}: {}", path, e)))?;
    // The manifest beside it records no identity for the V4 file now there
    let segment_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_prefix("vol_"))
        .and_then(|hex| u64::from_str_radix(hex, 16).ok())
        .ok_or_else(|| crate::core::Error::internal(format!("rewrite {:?}: no volume id", path)))?;
    let manifest_path = path.with_file_name("manifest.bin");
    let bytes = std::fs::read(&manifest_path)
        .map_err(|e| crate::core::Error::internal(format!("rewrite {:?}: {}", path, e)))?;
    let stored = super::manifest::TableManifest::deserialize(&bytes)
        .map_err(|e| crate::core::Error::internal(format!("rewrite {:?}: {}", path, e)))?;
    let mut manifest = stored.manifest;
    manifest.identities.remove(&segment_id);
    manifest.write_to_disk(&manifest_path, &stored.tombstones.into_iter().collect())
}

/// Read a frozen volume from disk, V5 or V4 by its magic.
pub fn read_volume_from_disk(path: &Path) -> Result<FrozenVolume> {
    // The V4 reader checks the format itself, so this opens the file once
    // and the store comes back holding the handle every later read shares
    let handle = super::writer::VolumeFile::shared(path);
    read_volume_from_handle(&handle)
}

/// Reads the volume a handle names, through that same handle. A compaction
/// that retires the file cannot take it from a holder, and the store that
/// comes back keeps it alive with it, so a reader that pinned the file can
/// still read a volume the manifest no longer lists.
pub fn read_volume_from_handle(
    handle: &std::sync::Arc<super::writer::VolumeFile>,
) -> Result<FrozenVolume> {
    use std::io::Seek;
    let open_error = |e| {
        crate::core::Error::internal(format!("failed to open volume {:?}: {}", handle.path(), e))
    };
    let mut file = handle.open().map_err(open_error)?;
    // A file shorter than a magic is refused by the V4 reader
    let mut magic = [0u8; 4];
    if super::writer::read_exact_at(&file, &mut magic, 0).is_ok() && magic == super::v5::V5_MAGIC {
        return read_volume_v5(file, std::sync::Arc::clone(handle), &handle.path());
    }
    // A positioned read moves the cursor on Windows and WASI; V4 reads from it
    file.rewind().map_err(open_error)?;
    let volume = read_volume_v4(file, std::sync::Arc::clone(handle), &handle.path())?;
    handle.mark_v4();
    Ok(volume)
}

/// List all volume files for a table, sorted by volume ID (oldest first).
pub fn list_volumes(dir: &Path, table_name: &str) -> Vec<PathBuf> {
    let table_dir = dir.join(table_name);
    let mut volumes: Vec<PathBuf> = match std::fs::read_dir(&table_dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e == VOLUME_EXT)
                    .unwrap_or(false)
            })
            .collect(),
        Err(_) => return Vec::new(),
    };
    volumes.sort(); // Sorted by filename = sorted by volume ID (hex)
    volumes
}

/// Load all volumes for a table from disk.
pub fn load_all_volumes(dir: &Path, table_name: &str) -> Result<Vec<Arc<FrozenVolume>>> {
    let paths = list_volumes(dir, table_name);
    let mut volumes = Vec::with_capacity(paths.len());
    for path in paths {
        let vol = read_volume_from_disk(&path)?;
        volumes.push(Arc::new(vol));
    }
    Ok(volumes)
}

/// Delete a specific volume file from disk.
pub fn delete_volume(path: &Path) -> Result<()> {
    super::writer::VolumeFile::retire_path(path).map_err(|e| {
        crate::core::Error::internal(format!("failed to delete volume {:?}: {}", path, e))
    })?;
    super::secondary::retire_side_of(path);
    Ok(())
}

/// Delete all volumes for a table, and the side files beside them.
pub fn delete_all_volumes(dir: &Path, table_name: &str) -> Result<()> {
    let paths = list_volumes(dir, table_name);
    let table_dir = dir.join(table_name);
    // The side files, of every generation, read once; those whose volume
    // is already gone go the same way
    let sides = super::secondary::SideFiles::in_dir(&table_dir);
    for path in paths {
        super::writer::VolumeFile::retire_path(&path).map_err(|e| {
            crate::core::Error::internal(format!("failed to delete volume {:?}: {}", path, e))
        })?;
    }
    sides.retire_all();
    // Remove the table directory if empty
    let _ = std::fs::remove_dir(&table_dir); // OK if not empty
    Ok(())
}

/// Generate a new volume ID. Monotonically increasing, unique across calls.
/// Uses microseconds since epoch + CAS loop for uniqueness.
pub fn next_volume_id() -> u64 {
    use crate::common::time_compat::{SystemTime, UNIX_EPOCH};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let micros = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;
    loop {
        let current = COUNTER.load(Ordering::Acquire);
        // New ID is at least micros, or current+1 if clock hasn't advanced
        let candidate = if micros > current {
            micros
        } else {
            current + 1
        };
        match COUNTER.compare_exchange_weak(current, candidate, Ordering::AcqRel, Ordering::Relaxed)
        {
            Ok(_) => return candidate,
            Err(_) => continue,
        }
    }
}

/// Simple volume catalog that tracks which volumes exist for each table.
///
/// This is a lightweight metadata file that allows the engine to know
/// which volumes to load without scanning the filesystem.
#[derive(Debug, Clone)]
pub struct VolumeCatalog {
    /// Volume entries per table: (volume_id, row_count, time_min_micros, time_max_micros)
    pub tables: ahash::AHashMap<String, Vec<VolumeEntry>>,
}

/// Metadata for a single volume.
#[derive(Debug, Clone)]
pub struct VolumeEntry {
    /// Unique volume identifier (timestamp-based)
    pub volume_id: u64,
    /// Number of rows in this volume
    pub row_count: u64,
    /// Minimum timestamp in micros (for time-range pruning without loading)
    pub time_min_micros: i64,
    /// Maximum timestamp in micros
    pub time_max_micros: i64,
}

impl VolumeCatalog {
    /// Create an empty catalog.
    pub fn new() -> Self {
        Self {
            tables: ahash::AHashMap::new(),
        }
    }

    /// Add a volume entry for a table.
    pub fn add_volume(&mut self, table_name: &str, entry: VolumeEntry) {
        self.tables
            .entry(table_name.to_string())
            .or_default()
            .push(entry);
    }

    /// Get volume entries for a table.
    pub fn get_volumes(&self, table_name: &str) -> &[VolumeEntry] {
        self.tables
            .get(table_name)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Serialize the catalog to bytes with trailing CRC32.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"STVC"); // SToolap Volume Catalog
        buf.extend_from_slice(&1u32.to_le_bytes()); // version

        let table_count = self.tables.len() as u32;
        buf.extend_from_slice(&table_count.to_le_bytes());

        for (name, entries) in &self.tables {
            let name_bytes = name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);

            buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for entry in entries {
                buf.extend_from_slice(&entry.volume_id.to_le_bytes());
                buf.extend_from_slice(&entry.row_count.to_le_bytes());
                buf.extend_from_slice(&entry.time_min_micros.to_le_bytes());
                buf.extend_from_slice(&entry.time_max_micros.to_le_bytes());
            }
        }
        // Trailing CRC32 for integrity validation on load
        let crc = crc32fast::hash(&buf);
        buf.extend_from_slice(&crc.to_le_bytes());
        buf
    }

    fn read_u32(data: &[u8], pos: &mut usize) -> std::io::Result<u32> {
        let end = *pos + 4;
        if end > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated volume catalog: expected u32",
            ));
        }
        let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
        *pos = end;
        Ok(v)
    }

    fn read_u64(data: &[u8], pos: &mut usize) -> std::io::Result<u64> {
        let end = *pos + 8;
        if end > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated volume catalog: expected u64",
            ));
        }
        let v = u64::from_le_bytes([
            data[*pos],
            data[*pos + 1],
            data[*pos + 2],
            data[*pos + 3],
            data[*pos + 4],
            data[*pos + 5],
            data[*pos + 6],
            data[*pos + 7],
        ]);
        *pos = end;
        Ok(v)
    }

    fn read_i64(data: &[u8], pos: &mut usize) -> std::io::Result<i64> {
        let end = *pos + 8;
        if end > data.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "truncated volume catalog: expected i64",
            ));
        }
        let v = i64::from_le_bytes([
            data[*pos],
            data[*pos + 1],
            data[*pos + 2],
            data[*pos + 3],
            data[*pos + 4],
            data[*pos + 5],
            data[*pos + 6],
            data[*pos + 7],
        ]);
        *pos = end;
        Ok(v)
    }

    /// Deserialize a catalog from bytes.
    pub fn deserialize(data: &[u8]) -> std::io::Result<Self> {
        // Minimum: magic(4) + version(4) + table_count(4) + crc(4) = 16
        if data.len() < 16 || &data[0..4] != b"STVC" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid volume catalog",
            ));
        }
        // Verify trailing CRC32
        let payload = &data[..data.len() - 4];
        let stored_crc = u32::from_le_bytes(data[data.len() - 4..].try_into().map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated catalog CRC")
        })?);
        let computed_crc = crc32fast::hash(payload);
        if stored_crc != computed_crc {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "volume catalog CRC mismatch: stored={:#x} computed={:#x}",
                    stored_crc, computed_crc
                ),
            ));
        }
        let mut pos = 4;

        let _version = Self::read_u32(data, &mut pos)?;
        let table_count = Self::read_u32(data, &mut pos)? as usize;

        let mut tables = ahash::AHashMap::new();

        for _ in 0..table_count {
            let name_len = Self::read_u32(data, &mut pos)? as usize;
            if pos + name_len > data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "truncated volume catalog: table name",
                ));
            }
            let name = std::str::from_utf8(&data[pos..pos + name_len])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
                .to_string();
            pos += name_len;

            let entry_count = Self::read_u32(data, &mut pos)? as usize;

            let mut entries = Vec::with_capacity(entry_count);
            for _ in 0..entry_count {
                let volume_id = Self::read_u64(data, &mut pos)?;
                let row_count = Self::read_u64(data, &mut pos)?;
                let time_min = Self::read_i64(data, &mut pos)?;
                let time_max = Self::read_i64(data, &mut pos)?;

                entries.push(VolumeEntry {
                    volume_id,
                    row_count,
                    time_min_micros: time_min,
                    time_max_micros: time_max,
                });
            }
            tables.insert(name, entries);
        }

        Ok(Self { tables })
    }

    /// Write catalog to disk atomically.
    pub fn write_to_disk(&self, dir: &Path) -> Result<()> {
        let data = self.serialize();
        let final_path = dir.join(CATALOG_FILE);
        let tmp_path = dir.join(format!("{}.tmp", CATALOG_FILE));

        // Write to tmp file and fsync BEFORE rename for crash safety.
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
                crate::core::Error::internal(format!("failed to create catalog tmp file: {}", e))
            })?;
            f.write_all(&data).map_err(|e| {
                crate::core::Error::internal(format!("failed to write volume catalog: {}", e))
            })?;
            f.sync_all().map_err(|e| {
                crate::core::Error::internal(format!("failed to fsync volume catalog: {}", e))
            })?;
        }

        std::fs::rename(&tmp_path, &final_path).map_err(|e| {
            crate::core::Error::internal(format!("failed to rename volume catalog: {}", e))
        })?;

        // Fsync directory to ensure the rename is durable.
        // Windows does not support opening directories for fsync;
        // NTFS metadata is flushed with the file's sync_all().
        #[cfg(not(windows))]
        {
            let d = std::fs::File::open(dir).map_err(|e| {
                std::io::Error::other(format!("failed to open dir for fsync: {}", e))
            })?;
            d.sync_all()
                .map_err(|e| std::io::Error::other(format!("failed to fsync dir: {}", e)))?;
        }

        Ok(())
    }

    /// Read catalog from disk.
    pub fn read_from_disk(dir: &Path) -> Result<Self> {
        let path = dir.join(CATALOG_FILE);
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = std::fs::read(&path).map_err(|e| {
            crate::core::Error::internal(format!("failed to read volume catalog: {}", e))
        })?;
        Self::deserialize(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to parse volume catalog: {}", e))
        })
    }
}

impl Default for VolumeCatalog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::writer::VolumeBuilder;
    use super::*;
    use crate::core::{DataType, Row, SchemaBuilder, Value};

    #[test]
    fn test_write_and_read_volume() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::text("hello")]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![Value::Integer(2), Value::text("world")]),
        );
        let vol = builder.finish().unwrap();

        let path = write_volume_to_disk(dir.path(), "test_table", 1, &vol).unwrap();
        assert!(path.exists());

        let loaded = read_volume_from_disk(&path).unwrap();
        assert_eq!(loaded.meta.row_count, 2);
        assert_eq!(loaded.columns.get(0).unwrap().get_i64(0), 1);
        assert_eq!(loaded.columns.get(1).unwrap().get_str(1), "world");
    }

    #[test]
    fn test_list_volumes() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        for i in 0..3 {
            let mut builder = VolumeBuilder::new(&schema);
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
            let vol = builder.finish().unwrap();
            write_volume_to_disk(dir.path(), "my_table", i as u64, &vol).unwrap();
        }

        let paths = list_volumes(dir.path(), "my_table");
        assert_eq!(paths.len(), 3);
    }

    #[test]
    fn test_load_all_volumes() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        for i in 1..=3 {
            let mut builder = VolumeBuilder::new(&schema);
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
            let vol = builder.finish().unwrap();
            write_volume_to_disk(dir.path(), "t", i as u64, &vol).unwrap();
        }

        let volumes = load_all_volumes(dir.path(), "t").unwrap();
        assert_eq!(volumes.len(), 3);
        assert_eq!(volumes[0].meta.row_count, 1);
    }

    #[test]
    fn test_catalog_roundtrip() {
        let mut catalog = VolumeCatalog::new();
        catalog.add_volume(
            "candlesticks_t1m",
            VolumeEntry {
                volume_id: 1000,
                row_count: 500_000,
                time_min_micros: 1_700_000_000_000_000,
                time_max_micros: 1_700_100_000_000_000,
            },
        );
        catalog.add_volume(
            "candlesticks_t1m",
            VolumeEntry {
                volume_id: 2000,
                row_count: 300_000,
                time_min_micros: 1_700_100_000_000_000,
                time_max_micros: 1_700_200_000_000_000,
            },
        );
        catalog.add_volume(
            "tickers",
            VolumeEntry {
                volume_id: 3000,
                row_count: 100,
                time_min_micros: 0,
                time_max_micros: 0,
            },
        );

        let data = catalog.serialize();
        let loaded = VolumeCatalog::deserialize(&data).unwrap();

        assert_eq!(loaded.get_volumes("candlesticks_t1m").len(), 2);
        assert_eq!(loaded.get_volumes("tickers").len(), 1);
        assert_eq!(loaded.get_volumes("nonexistent").len(), 0);
        assert_eq!(loaded.get_volumes("candlesticks_t1m")[0].row_count, 500_000);
    }

    #[test]
    fn test_catalog_disk_roundtrip() {
        let dir = tempfile::tempdir().unwrap();

        let mut catalog = VolumeCatalog::new();
        catalog.add_volume(
            "t1",
            VolumeEntry {
                volume_id: 42,
                row_count: 1000,
                time_min_micros: 100,
                time_max_micros: 200,
            },
        );

        catalog.write_to_disk(dir.path()).unwrap();
        let loaded = VolumeCatalog::read_from_disk(dir.path()).unwrap();

        assert_eq!(loaded.get_volumes("t1").len(), 1);
        assert_eq!(loaded.get_volumes("t1")[0].volume_id, 42);
    }

    #[test]
    fn test_delete_volumes() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Integer(1)]));
        let vol = builder.finish().unwrap();
        write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();

        assert_eq!(list_volumes(dir.path(), "t").len(), 1);
        delete_all_volumes(dir.path(), "t").unwrap();
        assert_eq!(list_volumes(dir.path(), "t").len(), 0);
    }

    #[test]
    fn test_v5_roundtrip_basic() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .column("price", DataType::Float, false, false)
            .build();

        let mut builder = VolumeBuilder::with_capacity(&schema, 3);
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
                Value::text("apple"),
                Value::Float(3.00),
            ]),
        );
        let vol = builder.finish().unwrap();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], b"STV5");
        assert_eq!(&bytes[bytes.len() - 4..], b"STV5");

        // Read back and verify eager loading
        let loaded = read_volume_from_disk(&path).unwrap();
        assert_eq!(loaded.meta.row_count, 3);

        // Access columns triggers decompression from RAM
        assert_eq!(loaded.columns.get(0).unwrap().get_i64(0), 1);
        assert_eq!(loaded.columns.get(0).unwrap().get_i64(2), 3);
        assert_eq!(loaded.columns.get(1).unwrap().get_str(0), "apple");
        assert_eq!(loaded.columns.get(1).unwrap().get_str(1), "banana");
        assert_eq!(loaded.columns.get(2).unwrap().get_f64(1), 0.75);

        // Zone maps survived
        assert_eq!(loaded.meta.zone_maps[0].min, Value::Integer(1));
        assert_eq!(loaded.meta.zone_maps[0].max, Value::Integer(3));

        // Stats survived
        assert_eq!(loaded.meta.stats.count_star(), 3);
        assert_eq!(loaded.meta.stats.sum(2), 5.25);

        // Sorted flags survived
        assert!(loaded.meta.sorted_columns[0]);

        // Row IDs survived
        assert_eq!(loaded.meta.row_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_roundtrip_with_nulls() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("value", DataType::Float, true, false)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(1), Value::Float(10.0)]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![Value::Integer(2), Value::Null(DataType::Float)]),
        );
        builder.add_row(
            3,
            &Row::from_values(vec![Value::Integer(3), Value::Float(30.0)]),
        );
        let vol = builder.finish().unwrap();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.meta.row_count, 3);
        assert!(!loaded.columns.get(1).unwrap().is_null(0));
        assert!(loaded.columns.get(1).unwrap().is_null(1));
        assert!(!loaded.columns.get(1).unwrap().is_null(2));
        assert_eq!(loaded.columns.get(1).unwrap().get_f64(0), 10.0);
        assert_eq!(loaded.columns.get(1).unwrap().get_f64(2), 30.0);
    }

    #[test]
    fn test_roundtrip_multiple_row_groups() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("label", DataType::Text, false, false)
            .build();

        // Create > ROW_GROUP_SIZE rows to exercise multi-group path
        let n = 70_000; // > 65536 (ROW_GROUP_SIZE)
        let mut builder = VolumeBuilder::with_capacity(&schema, n);
        for i in 0..n {
            builder.add_row(
                i as i64,
                &Row::from_values(vec![
                    Value::Integer(i as i64),
                    Value::text(if i % 2 == 0 { "even" } else { "odd" }),
                ]),
            );
        }
        let vol = builder.finish().unwrap();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.meta.row_count, n);

        // Check first, middle, and last rows
        assert_eq!(loaded.columns.get(0).unwrap().get_i64(0), 0);
        assert_eq!(
            loaded.columns.get(0).unwrap().get_i64(n / 2),
            (n / 2) as i64
        );
        assert_eq!(
            loaded.columns.get(0).unwrap().get_i64(n - 1),
            (n - 1) as i64
        );
        assert_eq!(loaded.columns.get(1).unwrap().get_str(0), "even");
        assert_eq!(loaded.columns.get(1).unwrap().get_str(1), "odd");
        assert_eq!(loaded.columns.get(1).unwrap().get_str(n - 1), "odd");

        // Row groups present
        assert!(!loaded.meta.row_groups.is_empty());
    }

    #[test]
    fn test_roundtrip_timestamp_boolean() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("time", DataType::Timestamp, false, false)
            .column("flag", DataType::Boolean, false, false)
            .build();

        let ts = chrono::Utc::now();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Timestamp(ts), Value::Boolean(true)]),
        );
        builder.add_row(
            2,
            &Row::from_values(vec![
                Value::Timestamp(ts + chrono::Duration::minutes(1)),
                Value::Boolean(false),
            ]),
        );
        let vol = builder.finish().unwrap();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.meta.row_count, 2);
        // Timestamp nanosecond precision
        if let Value::Timestamp(loaded_ts) = loaded.columns.get(0).unwrap().get_value(0) {
            assert_eq!(loaded_ts.timestamp_nanos_opt(), ts.timestamp_nanos_opt());
        } else {
            panic!("expected Timestamp");
        }
        assert!(loaded.columns.get(1).unwrap().get_bool(0));
        assert!(!loaded.columns.get(1).unwrap().get_bool(1));
    }

    #[test]
    fn test_get_row_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(
            1,
            &Row::from_values(vec![Value::Integer(42), Value::text("test")]),
        );
        let vol = builder.finish().unwrap();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        let row = loaded.get_row(0).unwrap();
        assert_eq!(row.get(0), Some(&Value::Integer(42)));
        assert_eq!(row.get(1), Some(&Value::text("test")));
    }

    fn mixed_volume(rows: usize) -> FrozenVolume {
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("label", DataType::Text, true, false)
            .column("x", DataType::Float, true, false)
            .build();
        let mut builder = VolumeBuilder::with_capacity(&schema, rows);
        for i in 0..rows {
            let label = match i % 3 {
                0 => Value::Null(DataType::Text),
                1 => Value::text("one"),
                _ => Value::text(format!("t{}", i % 7)),
            };
            builder.add_row(
                i as i64 * 2,
                &Row::from_values(vec![
                    Value::Integer(i as i64 * 2),
                    label,
                    Value::Float(i as f64),
                ]),
            );
        }
        builder.finish().unwrap()
    }

    #[test]
    fn a_v5_volume_reads_a_bloom_filter_or_builds_a_dictionary_on_first_use() {
        let dir = tempfile::tempdir().unwrap();
        let vol = mixed_volume(1_000);
        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let read = read_volume_from_disk(&path).unwrap();
        let blooms = &read.meta.bloom_filters;
        assert_eq!(blooms.memory_size(), 0, "bloom words read at open");
        let store = read.columns.compressed_store().unwrap();
        let at_open = store.memory_size();

        let h = super::super::column::ColumnBloomFilter::hash_value_static(&Value::Integer(4));
        assert!(blooms.might_contain_hash(0, h).unwrap());
        let written = vol.meta.bloom_filters.filter(0).unwrap().unwrap();
        assert_eq!(blooms.memory_size(), written.memory_size());
        assert_eq!(blooms.filter(0).unwrap().unwrap().bits(), written.bits());

        let entries = store.get_column_dictionary(1).unwrap();
        let want = vol.columns.get_column_dictionary(1).unwrap().unwrap();
        assert_eq!(&entries[..], &want[..]);
        assert!(store.memory_size() > at_open, "dictionary built at open");
    }

    #[test]
    fn readers_probing_a_column_at_once_read_its_bloom_words_once() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_volume_to_disk(dir.path(), "t", 1, &mixed_volume(200_000)).unwrap();
        let read = read_volume_from_disk(&path).unwrap();
        let blooms = &read.meta.bloom_filters;
        let h = super::super::column::ColumnBloomFilter::hash_value_static(&Value::Integer(4));
        let start = std::sync::Barrier::new(8);
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    start.wait();
                    assert!(blooms.might_contain_hash(0, h).unwrap());
                });
            }
        });
        assert_eq!(blooms.file_reads(0), 1);
    }

    #[test]
    fn the_shared_dictionary_text_counts_until_every_column_lets_it_go() {
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("long", DataType::Text, false, false)
            .column("short", DataType::Text, false, false)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        for i in 0..2_000i64 {
            let long = Value::text(format!("a distinct value long enough {i}"));
            let short = Value::text(format!("s{}", i % 3));
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i), long, short]));
        }
        let dir = tempfile::tempdir().unwrap();
        let path = write_volume_to_disk(dir.path(), "t", 1, &builder.finish().unwrap()).unwrap();
        let read = read_volume_from_disk(&path).unwrap();
        let store = read.columns.compressed_store().unwrap();
        let strings = |col| -> usize {
            let entries = store.get_column_dictionary(col).unwrap();
            entries.iter().map(|s| s.len() + 24).sum()
        };
        let text: usize = (0..2_000)
            .map(|i| format!("a distinct value long enough {i}").len())
            .sum();
        let long = strings(1);
        assert!(
            store.memory_size() >= long + text,
            "the unbuilt column keeps the whole text: {} < {}",
            store.memory_size(),
            long + text
        );
        let short = strings(2);
        assert_eq!(store.memory_size(), long + short, "the text is let go");
    }

    #[test]
    fn a_v4_file_reads_as_the_v5_file_of_the_same_volume() {
        let dir = tempfile::tempdir().unwrap();
        let rows = 70_000;
        let vol = mixed_volume(rows);
        let v4 = write_volume_to_disk_v4(dir.path(), "t", 1, &vol).unwrap();
        let v5 = write_volume_to_disk(dir.path(), "t", 2, &vol).unwrap();
        assert_eq!(&std::fs::read(&v4).unwrap()[..4], b"STV4");
        let (old, new) = (
            read_volume_from_disk(&v4).unwrap(),
            read_volume_from_disk(&v5).unwrap(),
        );
        let zones = |z: &[super::super::column::ZoneMap]| -> Vec<(Value, Value, u32, u32)> {
            z.iter()
                .map(|z| (z.min.clone(), z.max.clone(), z.null_count, z.row_count))
                .collect()
        };
        let groups = |v: &FrozenVolume| -> Vec<_> {
            v.meta
                .row_groups
                .iter()
                .map(|g| (g.start_idx, g.end_idx, zones(&g.zone_maps)))
                .collect()
        };
        assert_eq!(old.meta.row_ids, new.meta.row_ids);
        assert_eq!(old.meta.group_rows, new.meta.group_rows);
        assert_eq!(groups(&old), groups(&new));
        assert_eq!(zones(&old.meta.zone_maps), zones(&new.meta.zone_maps));
        for row in [0, 1, 2, 65_535, 65_536, rows - 1] {
            let want = vol.get_row(row).unwrap();
            assert_eq!(old.get_row(row).unwrap(), want, "row {row}");
            assert_eq!(new.get_row(row).unwrap(), want, "row {row}");
        }
    }

    #[test]
    fn the_column_directory_refuses_a_type_encoding_or_kind_it_lacks() {
        let vol = mixed_volume(10);
        let read = |patch: Option<(usize, u8)>| {
            let mut sections: Vec<(u32, u8, Vec<u8>)> =
                super::super::format::v5_volume_sections(&vol)
                    .unwrap()
                    .into_iter()
                    .map(|mut s| {
                        if let (super::super::format::V5_COLUMNS, Some((at, byte))) =
                            (s.kind, patch)
                        {
                            s.bytes[16 + at] = byte;
                        }
                        (s.kind, s.encoding, s.bytes)
                    })
                    .collect();
            super::super::format::v5_metadata_from_sections(&mut sections, &root_of(&vol, 10))
        };
        assert!(read(None).is_ok());
        // Type tag, encoding, kind and data type of the first column
        for (at, byte) in [(0, 0xEE), (3, 2), (4, 1), (5, 0xEE)] {
            assert!(read(Some((at, byte))).is_err(), "byte {at} = {byte:#x}");
        }
    }

    fn root_of(vol: &FrozenVolume, rows: u64) -> super::super::v5::Root {
        super::super::v5::Root {
            cols: vol.columns.len() as u32,
            groups: vol.meta.row_count.div_ceil(vol.meta.group_rows) as u32,
            group_rows: vol.meta.group_rows as u32,
            rows,
            volume_id: 1,
            nonce: 0,
        }
    }

    fn id_volume(ids: &[i64]) -> FrozenVolume {
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();
        let mut builder = VolumeBuilder::new(&schema);
        for &id in ids {
            builder.add_row(id, &Row::from_values(vec![Value::Integer(id)]));
        }
        builder.finish().unwrap()
    }

    #[test]
    fn row_ids_that_end_at_the_largest_key_read_back() {
        let dir = tempfile::tempdir().unwrap();
        for (n, ids) in [vec![i64::MAX], vec![i64::MAX - 2, i64::MAX - 1, i64::MAX]]
            .iter()
            .enumerate()
        {
            let path =
                write_volume_to_disk(dir.path(), "t", n as u64 + 1, &id_volume(ids)).unwrap();
            let loaded = read_volume_from_disk(&path).unwrap();
            assert_eq!(&loaded.meta.row_ids, ids);
        }
    }

    #[test]
    fn a_row_count_the_groups_do_not_cover_is_refused_before_ids_are_expanded() {
        let vol = id_volume(&[1, 2, 3]);
        let mut sections: Vec<(u32, u8, Vec<u8>)> = super::super::format::v5_volume_sections(&vol)
            .unwrap()
            .into_iter()
            .map(|mut s| {
                let claim = (1u64 << 62).to_le_bytes();
                match s.kind {
                    super::super::format::V5_COLUMNS => s.bytes[0..8].copy_from_slice(&claim),
                    super::super::format::V5_ROW_IDS => s.bytes[8..16].copy_from_slice(&claim),
                    _ => {}
                }
                (s.kind, s.encoding, s.bytes)
            })
            .collect();
        let root = root_of(&vol, 1 << 62);
        let Err(error) = super::super::format::v5_metadata_from_sections(&mut sections, &root)
        else {
            panic!("a row count past the groups was accepted");
        };
        assert!(error.to_string().contains("cover"), "{error}");
        let root = root_of(&vol, 3);
        let Err(error) = super::super::format::v5_metadata_from_sections(&mut sections, &root)
        else {
            panic!("a row count past the root's was accepted");
        };
        assert!(error.to_string().contains("root does not match"), "{error}");
    }

    #[test]
    fn a_volume_without_columns_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("test").build();
        let vol = VolumeBuilder::new(&schema).finish().unwrap();
        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();
        assert_eq!((loaded.columns.len(), loaded.meta.row_count), (0, 0));
    }
}

#[cfg(test)]
mod sync_tests {
    #[test]
    fn an_ordered_sync_takes_a_file_and_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f");
        std::fs::write(&path, b"x").unwrap();
        // Opened for writing, as every caller's file is: Windows refuses
        // to flush a read-only handle
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        super::sync_ordered(&f).unwrap();
        #[cfg(not(windows))]
        super::sync_ordered(&std::fs::File::open(dir.path()).unwrap()).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_file_system_without_a_barrier_gets_a_full_sync_and_an_io_error_stands() {
        assert!(super::barrier_unsupported(Some(libc::EINVAL)));
        assert!(super::barrier_unsupported(Some(libc::ENOTSUP)));
        assert!(super::barrier_unsupported(Some(libc::ENOTTY)));
        assert!(!super::barrier_unsupported(Some(libc::EIO)));
        assert!(!super::barrier_unsupported(None));
    }
}
