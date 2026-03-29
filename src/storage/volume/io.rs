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

use crate::common::SmartString;
use crate::core::Result;

use super::column::ColumnData;
use super::column::ROW_GROUP_SIZE;
use super::format::{
    deserialize_column_block, deserialize_column_block_into, deserialize_volume,
    deserialize_volume_metadata, serialize_volume, serialize_volume_metadata, COL_BOOLEAN,
    COL_BYTES, COL_DICTIONARY, COL_FLOAT64, COL_INT64, COL_TIMESTAMP,
};
use super::writer::{CompressedBlockStore, FrozenVolume, LazyColumns};

/// Volume file extension
const VOLUME_EXT: &str = "vol";

/// Magic bytes for LZ4-compressed volumes. Uncompressed volumes start with
/// b"STVL" (the format magic). On read we check the first 4 bytes: STVZ
/// means decompress first, STVL means read directly. This keeps compression
/// fully transparent to the format module.
const COMPRESSED_MAGIC: [u8; 4] = *b"STVZ";

/// Magic bytes for V4 per-column per-group compressed format.
const V4_MAGIC: [u8; 4] = *b"STV4";

/// V4 format version. Bump when the metadata or block layout changes.
const V4_VERSION: u32 = 1;

/// Volume catalog filename
const CATALOG_FILE: &str = "volumes.catalog";

/// Write a frozen volume to disk atomically.
///
/// Writes to a .tmp file first, then renames to the final name.
/// This prevents partial files from crashes during writes.
pub fn write_volume_to_disk(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
) -> Result<PathBuf> {
    let (path, _store) = write_volume_to_disk_opts(dir, table_name, volume_id, volume, true)?;
    Ok(path)
}

/// Write a frozen volume to disk atomically, with optional LZ4 compression.
///
/// When `compress` is true, writes V4 format (per-column per-group LZ4 blocks).
/// V4 enables lazy column loading on read — only metadata is decompressed at
/// startup, columns are decompressed from RAM on demand.
/// When false, writes legacy STVL format.
/// Returns (path, Option<CompressedBlockStore>). When V4, the store is returned
/// so callers can register a lazy volume without re-reading from disk.
pub fn write_volume_to_disk_opts(
    dir: &Path,
    table_name: &str,
    volume_id: u64,
    volume: &FrozenVolume,
    compress: bool,
) -> Result<(PathBuf, Option<CompressedBlockStore>)> {
    let table_dir = dir.join(table_name);
    std::fs::create_dir_all(&table_dir)
        .map_err(|e| crate::core::Error::internal(format!("failed to create volume dir: {}", e)))?;

    let filename = format!("vol_{:016x}.{}", volume_id, VOLUME_EXT);
    let final_path = table_dir.join(&filename);
    let tmp_path = table_dir.join(format!("{}.tmp", filename));

    let (data, store) = if compress {
        let (bytes, s) = serialize_v4(volume)
            .map_err(|e| crate::core::Error::internal(format!("V4 serialize failed: {}", e)))?;
        (bytes, Some(s))
    } else {
        let bytes = serialize_volume(volume)
            .map_err(|e| crate::core::Error::internal(format!("serialize failed: {}", e)))?;
        (bytes, None)
    };

    {
        use std::io::Write;
        let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
            crate::core::Error::internal(format!("failed to create volume tmp file: {}", e))
        })?;
        f.write_all(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to write volume file: {}", e))
        })?;
        f.sync_all().map_err(|e| {
            crate::core::Error::internal(format!("failed to fsync volume tmp file: {}", e))
        })?;
    }
    drop(data);

    std::fs::rename(&tmp_path, &final_path).map_err(|e| {
        crate::core::Error::internal(format!("failed to rename volume file: {}", e))
    })?;

    #[cfg(not(windows))]
    if let Ok(d) = std::fs::File::open(&table_dir) {
        d.sync_all().map_err(|e| {
            crate::core::Error::internal(format!("failed to fsync volume directory: {}", e))
        })?;
    }

    Ok((final_path, store))
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
/// Returns (file_bytes, CompressedBlockStore). The store can be used to register
/// a lazy volume without re-reading from disk — avoids keeping eager columns in RAM.
fn serialize_v4(vol: &FrozenVolume) -> std::io::Result<(Vec<u8>, CompressedBlockStore)> {
    use std::io::Write;

    let col_count = vol.columns.len();
    let group_size = ROW_GROUP_SIZE;
    let num_groups = if vol.row_count == 0 {
        0
    } else {
        vol.row_count.div_ceil(group_size)
    };

    // 1. Serialize + compress metadata
    let meta_raw = serialize_volume_metadata(vol)?;
    let meta_compressed = lz4_flex::compress_prepend_size(&meta_raw);
    drop(meta_raw);

    // 2. Build CompressedBlockStore (compresses all column blocks)
    let store =
        CompressedBlockStore::compress_columns(&vol.columns, &vol.column_types, vol.row_count);

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

/// Read a V4 volume via streaming I/O. Never holds the full file in memory.
/// CRC32 is computed incrementally as sections are read.
/// Blocks are read one at a time into a reusable buffer and decompressed
/// directly into final column vectors. No intermediate compressed storage.
fn read_volume_v4(path: &Path) -> Result<FrozenVolume> {
    use std::io::Read;

    let inv = |msg: &str| crate::core::Error::internal(format!("V4: {}", msg));

    let file = std::fs::File::open(path)
        .map_err(|e| crate::core::Error::internal(format!("V4 open {:?}: {}", path, e)))?;
    let file_len = file
        .metadata()
        .map_err(|e| crate::core::Error::internal(format!("V4 stat {:?}: {}", path, e)))?
        .len() as usize;
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

    // 2. Compressed metadata (read into temp buffer, decompress, drop)
    let mut meta_compressed = vec![0u8; meta_len];
    crc_read!(&mut meta_compressed);

    // Parse prepended uncompressed size (4 bytes LE), then decompress_into
    // to avoid lz4_flex::decompress_size_prepended allocating a fresh Vec.
    let meta_raw = if meta_compressed.len() >= 4 {
        let uncomp_size = u32::from_le_bytes(meta_compressed[..4].try_into().unwrap()) as usize;
        let mut buf = vec![0u8; uncomp_size];
        lz4_flex::decompress_into(&meta_compressed[4..], &mut buf)
            .map_err(|e| inv(&format!("metadata LZ4: {}", e)))?;
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

    // 3. Block index: (compressed_len: u64, decompressed_len: u64) pairs
    let total_blocks = col_count * num_groups;
    let mut index_buf = vec![0u8; total_blocks * 16];
    crc_read!(&mut index_buf);

    let mut compressed_lens = Vec::with_capacity(total_blocks);
    let mut decompressed_lens_flat = Vec::with_capacity(total_blocks);
    for i in 0..total_blocks {
        let off = i * 16;
        compressed_lens
            .push(u64::from_le_bytes(index_buf[off..off + 8].try_into().unwrap()) as usize);
        decompressed_lens_flat
            .push(u64::from_le_bytes(index_buf[off + 8..off + 16].try_into().unwrap()) as usize);
    }
    drop(index_buf);

    // 4. Build per-column dictionaries (needed before streaming decompression)
    let mut col_dicts: Vec<Option<Arc<[SmartString]>>> = vec![None; col_count];
    {
        let mut dict_start = 0usize;
        for (i, &tag) in meta.col_type_tags.iter().enumerate() {
            if tag == COL_DICTIONARY {
                let count = meta.col_dict_counts[i] as usize;
                let end = dict_start + count;
                col_dicts[i] = Some(Arc::from(&meta.shared_dict[dict_start..end]));
                dict_start = end;
            }
        }
    }

    let col_data_types = meta.column_types.clone();
    let row_count = meta.row_count;
    let group_size = ROW_GROUP_SIZE;

    // 5. Stream-decompress: read each block into a reusable buffer, decompress
    //    directly into pre-allocated column vectors. No CompressedBlockStore,
    //    no intermediate Vec<Vec<Vec<u8>>> — saves ~1.3 GB allocation throughput.
    let mut read_buf: Vec<u8> = Vec::new(); // reusable disk-read buffer
    let mut lz4_buf: Vec<u8> = Vec::new(); // reusable LZ4 scratch buffer
    let mut block_idx = 0;

    let mut eager_columns: Vec<ColumnData> = Vec::with_capacity(col_count);

    #[allow(clippy::needless_range_loop)]
    for col_idx in 0..col_count {
        let type_tag = meta.col_type_tags[col_idx];
        let ext_type = crate::core::DataType::from_u8(meta.col_ext_types[col_idx])
            .unwrap_or(crate::core::DataType::Null);

        if num_groups == 0 {
            // Empty volume — push empty column data
            let col = match type_tag {
                COL_INT64 => ColumnData::Int64 {
                    values: Vec::new(),
                    nulls: Vec::new(),
                },
                COL_FLOAT64 => ColumnData::Float64 {
                    values: Vec::new(),
                    nulls: Vec::new(),
                },
                COL_TIMESTAMP => ColumnData::TimestampNanos {
                    values: Vec::new(),
                    nulls: Vec::new(),
                },
                COL_BOOLEAN => ColumnData::Boolean {
                    values: Vec::new(),
                    nulls: Vec::new(),
                },
                COL_DICTIONARY => ColumnData::Dictionary {
                    ids: Vec::new(),
                    dictionary: col_dicts[col_idx]
                        .take()
                        .unwrap_or_else(|| Arc::from(Vec::<SmartString>::new())),
                    nulls: Vec::new(),
                },
                COL_BYTES => ColumnData::Bytes {
                    data: Vec::new(),
                    offsets: Vec::new(),
                    ext_type,
                    nulls: Vec::new(),
                },
                _ => return Err(inv(&format!("unknown col type tag {}", type_tag))),
            };
            eager_columns.push(col);
        } else if num_groups == 1 {
            // Single group: read block, decompress, deserialize via existing path
            let comp_len = compressed_lens[block_idx];
            let decomp_len = decompressed_lens_flat[block_idx];
            read_buf.clear();
            read_buf.resize(comp_len, 0);
            crc_read!(&mut read_buf);
            block_idx += 1;

            let group_rows = row_count.min(group_size);
            let dict = col_dicts[col_idx].as_ref().map(Arc::clone);

            let col = if read_buf.len() == decomp_len {
                // Stored uncompressed
                deserialize_column_block(&read_buf, type_tag, group_rows, dict, ext_type)
                    .map_err(|e| inv(&format!("col={}: {}", col_idx, e)))?
            } else {
                if lz4_buf.len() < decomp_len {
                    lz4_buf.resize(decomp_len, 0);
                }
                lz4_flex::decompress_into(&read_buf, &mut lz4_buf[..decomp_len])
                    .map_err(|e| inv(&format!("col={} LZ4: {}", col_idx, e)))?;
                deserialize_column_block(
                    &lz4_buf[..decomp_len],
                    type_tag,
                    group_rows,
                    dict,
                    ext_type,
                )
                .map_err(|e| inv(&format!("col={}: {}", col_idx, e)))?
            };
            eager_columns.push(col);
        } else {
            // Multi-group: stream blocks directly into pre-allocated vectors.
            // Helper closure: compute row count for a group index.
            let group_row_count = |gi: usize| -> usize {
                if gi == num_groups - 1 {
                    row_count - gi * group_size
                } else {
                    group_size
                }
            };

            // Read + decompress each group block inline into typed output vecs.
            macro_rules! stream_decompress_into {
                ($nulls:expr, $i64_out:expr, $f64_out:expr, $u32_out:expr,
                 $bool_out:expr, $bytes_data:expr, $bytes_offsets:expr) => {
                    for gi in 0..num_groups {
                        let comp_len = compressed_lens[block_idx];
                        let decomp_len = decompressed_lens_flat[block_idx];
                        read_buf.clear();
                        read_buf.resize(comp_len, 0);
                        crc_read!(&mut read_buf);
                        block_idx += 1;

                        let grp_rows = group_row_count(gi);
                        if read_buf.len() == decomp_len {
                            // Stored uncompressed — deserialize directly
                            deserialize_column_block_into(
                                &read_buf,
                                type_tag,
                                grp_rows,
                                $nulls,
                                $i64_out,
                                $f64_out,
                                $u32_out,
                                $bool_out,
                                $bytes_data,
                                $bytes_offsets,
                            )
                            .map_err(|e| inv(&format!("col={} g={}: {}", col_idx, gi, e)))?;
                        } else {
                            if lz4_buf.len() < decomp_len {
                                lz4_buf.resize(decomp_len, 0);
                            }
                            lz4_flex::decompress_into(&read_buf, &mut lz4_buf[..decomp_len])
                                .map_err(|e| {
                                    inv(&format!("col={} g={} LZ4: {}", col_idx, gi, e))
                                })?;
                            deserialize_column_block_into(
                                &lz4_buf[..decomp_len],
                                type_tag,
                                grp_rows,
                                $nulls,
                                $i64_out,
                                $f64_out,
                                $u32_out,
                                $bool_out,
                                $bytes_data,
                                $bytes_offsets,
                            )
                            .map_err(|e| inv(&format!("col={} g={}: {}", col_idx, gi, e)))?;
                        }
                    }
                };
            }

            let col = match type_tag {
                COL_INT64 => {
                    let mut vals = Vec::with_capacity(row_count);
                    let mut nulls = Vec::with_capacity(row_count);
                    stream_decompress_into!(
                        &mut nulls,
                        Some(&mut vals),
                        None,
                        None,
                        None,
                        None,
                        None
                    );
                    ColumnData::Int64 {
                        values: vals,
                        nulls,
                    }
                }
                COL_FLOAT64 => {
                    let mut vals = Vec::with_capacity(row_count);
                    let mut nulls = Vec::with_capacity(row_count);
                    stream_decompress_into!(
                        &mut nulls,
                        None,
                        Some(&mut vals),
                        None,
                        None,
                        None,
                        None
                    );
                    ColumnData::Float64 {
                        values: vals,
                        nulls,
                    }
                }
                COL_TIMESTAMP => {
                    let mut vals = Vec::with_capacity(row_count);
                    let mut nulls = Vec::with_capacity(row_count);
                    stream_decompress_into!(
                        &mut nulls,
                        Some(&mut vals),
                        None,
                        None,
                        None,
                        None,
                        None
                    );
                    ColumnData::TimestampNanos {
                        values: vals,
                        nulls,
                    }
                }
                COL_BOOLEAN => {
                    let mut vals = Vec::with_capacity(row_count);
                    let mut nulls = Vec::with_capacity(row_count);
                    stream_decompress_into!(
                        &mut nulls,
                        None,
                        None,
                        None,
                        Some(&mut vals),
                        None,
                        None
                    );
                    ColumnData::Boolean {
                        values: vals,
                        nulls,
                    }
                }
                COL_DICTIONARY => {
                    let mut ids = Vec::with_capacity(row_count);
                    let mut nulls = Vec::with_capacity(row_count);
                    stream_decompress_into!(
                        &mut nulls,
                        None,
                        None,
                        Some(&mut ids),
                        None,
                        None,
                        None
                    );
                    ColumnData::Dictionary {
                        ids,
                        dictionary: col_dicts[col_idx]
                            .take()
                            .unwrap_or_else(|| Arc::from(Vec::<SmartString>::new())),
                        nulls,
                    }
                }
                COL_BYTES => {
                    let mut data = Vec::new();
                    let mut offsets = Vec::with_capacity(row_count);
                    let mut nulls = Vec::with_capacity(row_count);
                    stream_decompress_into!(
                        &mut nulls,
                        None,
                        None,
                        None,
                        None,
                        Some(&mut data),
                        Some(&mut offsets)
                    );
                    ColumnData::Bytes {
                        data,
                        offsets,
                        ext_type,
                        nulls,
                    }
                }
                _ => return Err(inv(&format!("unknown col type tag {}", type_tag))),
            };
            eager_columns.push(col);
        }
    }

    // 6. Verify CRC32 (computed incrementally over everything we read)
    let mut crc_buf = [0u8; 4];
    reader
        .read_exact(&mut crc_buf)
        .map_err(|e| crate::core::Error::internal(format!("V4 CRC read: {}", e)))?;
    let stored_crc = u32::from_le_bytes(crc_buf);
    if hasher.finalize() != stored_crc {
        return Err(inv("CRC mismatch"));
    }

    Ok(FrozenVolume {
        columns: LazyColumns::eager(eager_columns, col_data_types),
        zone_maps: meta.zone_maps,
        bloom_filters: meta.bloom_filters,
        stats: meta.stats,
        row_count: meta.row_count,
        column_names: meta.column_names,
        column_types: meta.column_types,
        row_ids: meta.row_ids,
        sorted_columns: meta.col_sorted,
        column_name_map: meta.column_name_map,
        unique_indices: parking_lot::RwLock::new(rustc_hash::FxHashMap::default()),
        row_groups: meta.row_groups,
    })
}

/// Read a frozen volume from disk.
///
/// Detects format by peeking at magic bytes:
/// - STV4: streaming read with incremental CRC (no full file buffer)
/// - STVZ/STVL: legacy full read (converted to V4 on compaction)
pub fn read_volume_from_disk(path: &Path) -> Result<FrozenVolume> {
    use std::io::Read;

    let mut magic = [0u8; 4];
    {
        let mut f = std::fs::File::open(path).map_err(|e| {
            crate::core::Error::internal(format!("failed to open volume {:?}: {}", path, e))
        })?;
        f.read_exact(&mut magic).map_err(|e| {
            crate::core::Error::internal(format!("failed to read magic {:?}: {}", path, e))
        })?;
    }

    if magic == V4_MAGIC {
        read_volume_v4(path)
    } else {
        let data = std::fs::read(path).map_err(|e| {
            crate::core::Error::internal(format!("failed to read volume {:?}: {}", path, e))
        })?;
        if data.len() >= 4 && data[..4] == COMPRESSED_MAGIC {
            let raw = lz4_flex::decompress_size_prepended(&data[4..]).map_err(|e| {
                crate::core::Error::internal(format!("failed to decompress {:?}: {}", path, e))
            })?;
            deserialize_volume(&raw).map_err(|e| {
                crate::core::Error::internal(format!("failed to deserialize {:?}: {}", path, e))
            })
        } else {
            deserialize_volume(&data).map_err(|e| {
                crate::core::Error::internal(format!("failed to deserialize {:?}: {}", path, e))
            })
        }
    }
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
    std::fs::remove_file(path).map_err(|e| {
        crate::core::Error::internal(format!("failed to delete volume {:?}: {}", path, e))
    })
}

/// Delete all volumes for a table.
pub fn delete_all_volumes(dir: &Path, table_name: &str) -> Result<()> {
    let paths = list_volumes(dir, table_name);
    for path in paths {
        delete_volume(&path)?;
    }
    // Remove the table directory if empty
    let table_dir = dir.join(table_name);
    let _ = std::fs::remove_dir(&table_dir); // OK if not empty
    Ok(())
}

/// Generate a new volume ID. Monotonically increasing, unique across calls.
/// Uses microseconds since epoch + CAS loop for uniqueness.
pub fn next_volume_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
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
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "test_table", 1, &vol).unwrap();
        assert!(path.exists());

        let loaded = read_volume_from_disk(&path).unwrap();
        assert_eq!(loaded.row_count, 2);
        assert_eq!(loaded.columns[0].get_i64(0), 1);
        assert_eq!(loaded.columns[1].get_str(1), "world");
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
            let vol = builder.finish();
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
            let vol = builder.finish();
            write_volume_to_disk(dir.path(), "t", i as u64, &vol).unwrap();
        }

        let volumes = load_all_volumes(dir.path(), "t").unwrap();
        assert_eq!(volumes.len(), 3);
        assert_eq!(volumes[0].row_count, 1);
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
        let vol = builder.finish();
        write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();

        assert_eq!(list_volumes(dir.path(), "t").len(), 1);
        delete_all_volumes(dir.path(), "t").unwrap();
        assert_eq!(list_volumes(dir.path(), "t").len(), 0);
    }

    #[test]
    fn test_v4_roundtrip_basic() {
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
        let vol = builder.finish();

        // write_volume_to_disk with compress=true produces V4
        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        // Verify STV4 magic
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..4], b"STV4");

        // Read back and verify eager loading
        let loaded = read_volume_from_disk(&path).unwrap();
        assert_eq!(loaded.row_count, 3);

        // Access columns triggers decompression from RAM
        assert_eq!(loaded.columns[0].get_i64(0), 1);
        assert_eq!(loaded.columns[0].get_i64(2), 3);
        assert_eq!(loaded.columns[1].get_str(0), "apple");
        assert_eq!(loaded.columns[1].get_str(1), "banana");
        assert_eq!(loaded.columns[2].get_f64(1), 0.75);

        // Zone maps survived
        assert_eq!(loaded.zone_maps[0].min, Value::Integer(1));
        assert_eq!(loaded.zone_maps[0].max, Value::Integer(3));

        // Stats survived
        assert_eq!(loaded.stats.count_star(), 3);
        assert_eq!(loaded.stats.sum(2), 5.25);

        // Sorted flags survived
        assert!(loaded.sorted_columns[0]);

        // Row IDs survived
        assert_eq!(loaded.row_ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_v4_roundtrip_with_nulls() {
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
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.row_count, 3);
        assert!(!loaded.columns[1].is_null(0));
        assert!(loaded.columns[1].is_null(1));
        assert!(!loaded.columns[1].is_null(2));
        assert_eq!(loaded.columns[1].get_f64(0), 10.0);
        assert_eq!(loaded.columns[1].get_f64(2), 30.0);
    }

    #[test]
    fn test_v4_roundtrip_multiple_row_groups() {
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
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.row_count, n);

        // Check first, middle, and last rows
        assert_eq!(loaded.columns[0].get_i64(0), 0);
        assert_eq!(loaded.columns[0].get_i64(n / 2), (n / 2) as i64);
        assert_eq!(loaded.columns[0].get_i64(n - 1), (n - 1) as i64);
        assert_eq!(loaded.columns[1].get_str(0), "even");
        assert_eq!(loaded.columns[1].get_str(1), "odd");
        assert_eq!(loaded.columns[1].get_str(n - 1), "odd");

        // Row groups present
        assert!(!loaded.row_groups.is_empty());
    }

    #[test]
    fn test_v4_roundtrip_timestamp_boolean() {
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
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        assert_eq!(loaded.row_count, 2);
        // Timestamp nanosecond precision
        if let Value::Timestamp(loaded_ts) = loaded.columns[0].get_value(0) {
            assert_eq!(loaded_ts.timestamp_nanos_opt(), ts.timestamp_nanos_opt());
        } else {
            panic!("expected Timestamp");
        }
        assert!(loaded.columns[1].get_bool(0));
        assert!(!loaded.columns[1].get_bool(1));
    }

    #[test]
    fn test_v4_get_row_roundtrip() {
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
        let vol = builder.finish();

        let path = write_volume_to_disk(dir.path(), "t", 1, &vol).unwrap();
        let loaded = read_volume_from_disk(&path).unwrap();

        let row = loaded.get_row(0);
        assert_eq!(row.get(0), Some(&Value::Integer(42)));
        assert_eq!(row.get(1), Some(&Value::text("test")));
    }
}
