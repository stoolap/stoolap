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

//! Volume file format: serialization and deserialization of frozen volumes.
//!
//! File Layout:
//! ```text
//! ┌──────────────────────────────┐
//! │ Header (32 bytes)            │  magic, version, row_count, col_count
//! ├──────────────────────────────┤
//! │ Column Directory (N entries) │  type, data_offset, data_len, flags
//! ├──────────────────────────────┤
//! │ Column 0: null bitmap        │  1 byte per row (0=value, 1=null)
//! │ Column 0: typed data         │  raw bytes (i64/f64/u32/bool per row)
//! ├──────────────────────────────┤
//! │ Column 1: null bitmap + data │
//! ├──────────────────────────────┤
//! │ ...                          │
//! ├──────────────────────────────┤
//! │ String Dictionary            │  count + [len, bytes] per entry
//! ├──────────────────────────────┤
//! │ Row IDs                      │  i64 per row
//! ├──────────────────────────────┤
//! │ Zone Maps                    │  serialized min/max per column
//! ├──────────────────────────────┤
//! │ Stats                        │  pre-computed aggregates
//! └──────────────────────────────┘
//! ```

use std::io::{self, Write};

use crate::common::SmartString;
use crate::core::{DataType, Value};

use super::column::{ColumnData, ZoneMap};
use super::stats::{ColumnAggregateStats, VolumeAggregateStats};
use super::writer::FrozenVolume;

// Magic bytes: "STVL" (SToolap VoLume)
const MAGIC: [u8; 4] = *b"STVL";
const FORMAT_VERSION: u32 = 3;
const FORMAT_VERSION_V2: u32 = 2;

// Column type tags for the directory
const COL_INT64: u8 = 1;
const COL_FLOAT64: u8 = 2;
const COL_TIMESTAMP: u8 = 3;
const COL_BOOLEAN: u8 = 4;
const COL_DICTIONARY: u8 = 5;
const COL_BYTES: u8 = 6;

// Column flags
const FLAG_SORTED: u8 = 0x01;

/// Write a FrozenVolume to a byte buffer.
///
/// Returns the serialized bytes. The caller decides whether to write
/// to a file, mmap, or keep in memory (WASM).
pub fn serialize_volume(vol: &FrozenVolume) -> io::Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(vol.memory_size());

    // Header: magic(4) + version(4) + row_count(8) + col_count(4) + reserved(12)
    buf.write_all(&MAGIC)?;
    buf.write_all(&FORMAT_VERSION.to_le_bytes())?;
    buf.write_all(&(vol.row_count as u64).to_le_bytes())?;
    buf.write_all(&(vol.columns.len() as u32).to_le_bytes())?;
    buf.write_all(&[0u8; 12])?; // reserved

    // Collect all dictionaries from all dictionary columns into one shared dict
    let mut shared_dict: Vec<SmartString> = Vec::new();
    let mut dict_offsets: Vec<u32> = Vec::new(); // per dictionary column: start index in shared_dict

    for col in &vol.columns {
        if let ColumnData::Dictionary { dictionary, .. } = col {
            dict_offsets.push(shared_dict.len() as u32);
            shared_dict.extend(dictionary.iter().cloned());
        }
    }

    // Column directory: per column write type(1) + flags(1) + ext_or_dict(4)
    // flags byte: bit 0 = sorted, bits 1-7 reserved (NOT used for ext_type)
    let mut dict_col_idx = 0usize;
    for (i, col) in vol.columns.iter().enumerate() {
        let type_tag = match col {
            ColumnData::Int64 { .. } => COL_INT64,
            ColumnData::Float64 { .. } => COL_FLOAT64,
            ColumnData::TimestampNanos { .. } => COL_TIMESTAMP,
            ColumnData::Boolean { .. } => COL_BOOLEAN,
            ColumnData::Dictionary { .. } => COL_DICTIONARY,
            ColumnData::Bytes { .. } => COL_BYTES,
        };
        let sorted_flag = if vol.sorted_columns[i] {
            FLAG_SORTED
        } else {
            0
        };
        buf.push(type_tag);
        buf.push(sorted_flag);
        // 4-byte extra field: dict offset for Dictionary, ext_type for Bytes
        if type_tag == COL_DICTIONARY {
            buf.write_all(&dict_offsets[dict_col_idx].to_le_bytes())?;
            dict_col_idx += 1;
        } else if type_tag == COL_BYTES {
            let ext = match col {
                ColumnData::Bytes { ext_type, .. } => *ext_type as u32,
                _ => 0,
            };
            buf.write_all(&ext.to_le_bytes())?;
        } else {
            buf.write_all(&[0u8; 4])?; // padding
        }
    }

    // Column data: null bitmap + typed data for each column
    for col in &vol.columns {
        match col {
            ColumnData::Int64 { values, nulls } => {
                write_nulls(&mut buf, nulls)?;
                write_i64_bulk(&mut buf, values);
            }
            ColumnData::Float64 { values, nulls } => {
                write_nulls(&mut buf, nulls)?;
                write_f64_bulk(&mut buf, values);
            }
            ColumnData::TimestampNanos { values, nulls } => {
                write_nulls(&mut buf, nulls)?;
                write_i64_bulk(&mut buf, values);
            }
            ColumnData::Boolean { values, nulls } => {
                write_nulls(&mut buf, nulls)?;
                write_bool_bulk(&mut buf, values);
            }
            ColumnData::Dictionary { ids, nulls, .. } => {
                write_nulls(&mut buf, nulls)?;
                write_u32_bulk(&mut buf, ids);
            }
            ColumnData::Bytes {
                data,
                offsets,
                nulls,
                ..
            } => {
                write_nulls(&mut buf, nulls)?;
                // Write offsets array
                buf.write_all(&(offsets.len() as u64).to_le_bytes())?;
                for (off, len) in offsets {
                    buf.write_all(&off.to_le_bytes())?;
                    buf.write_all(&len.to_le_bytes())?;
                }
                // Write data blob
                buf.write_all(&(data.len() as u64).to_le_bytes())?;
                buf.write_all(data)?;
            }
        }
    }

    // Shared string dictionary
    buf.write_all(&(shared_dict.len() as u32).to_le_bytes())?;
    for s in &shared_dict {
        let bytes = s.as_bytes();
        buf.write_all(&(bytes.len() as u32).to_le_bytes())?;
        buf.write_all(bytes)?;
    }

    // Row IDs
    for id in &vol.row_ids {
        buf.write_all(&id.to_le_bytes())?;
    }

    // Zone maps
    for zm in &vol.zone_maps {
        write_value(&mut buf, &zm.min)?;
        write_value(&mut buf, &zm.max)?;
        buf.write_all(&zm.null_count.to_le_bytes())?;
        buf.write_all(&zm.row_count.to_le_bytes())?;
    }

    // Bloom filters (format v2+)
    buf.write_all(&(vol.bloom_filters.len() as u32).to_le_bytes())?;
    for bf in &vol.bloom_filters {
        buf.write_all(&(bf.num_bits() as u64).to_le_bytes())?;
        let data_bytes = bf.bits_as_bytes();
        buf.write_all(&(data_bytes.len() as u32).to_le_bytes())?;
        buf.write_all(&data_bytes)?;
    }

    // Stats
    buf.write_all(&(vol.stats.total_rows).to_le_bytes())?;
    buf.write_all(&(vol.stats.live_rows).to_le_bytes())?;
    buf.write_all(&(vol.stats.columns.len() as u32).to_le_bytes())?;
    for cs in &vol.stats.columns {
        buf.write_all(&cs.sum_int.to_le_bytes())?;
        buf.write_all(&cs.sum_float.to_le_bytes())?;
        buf.write_all(&cs.numeric_count.to_le_bytes())?;
        buf.write_all(&cs.non_null_count.to_le_bytes())?;
        write_value(&mut buf, &cs.min)?;
        write_value(&mut buf, &cs.max)?;
    }

    // Column names (for schema validation on load)
    for name in &vol.column_names {
        let bytes = name.as_bytes();
        buf.write_all(&(bytes.len() as u32).to_le_bytes())?;
        buf.write_all(bytes)?;
    }

    // Column types
    for dt in &vol.column_types {
        buf.push(*dt as u8);
    }

    // Row groups (v3+): per-group zone maps for sub-volume pruning
    buf.write_all(&(vol.row_groups.len() as u32).to_le_bytes())?;
    for rg in &vol.row_groups {
        buf.write_all(&rg.start_idx.to_le_bytes())?;
        buf.write_all(&rg.end_idx.to_le_bytes())?;
        for zm in &rg.zone_maps {
            write_value(&mut buf, &zm.min)?;
            write_value(&mut buf, &zm.max)?;
            buf.write_all(&zm.null_count.to_le_bytes())?;
            buf.write_all(&zm.row_count.to_le_bytes())?;
        }
    }

    // Trailing CRC32 over the entire payload (detects bitflips and partial writes)
    let crc = crc32fast::hash(&buf);
    buf.write_all(&crc.to_le_bytes())?;

    Ok(buf)
}

/// Deserialize a FrozenVolume from bytes.
///
/// Returns InvalidData on corrupted files. All reads are bounds-checked
/// so this never panics, even with panic=abort.
pub fn deserialize_volume(data: &[u8]) -> io::Result<FrozenVolume> {
    let mut pos = 0;

    // Header
    if data.len() < 36 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "volume too small",
        ));
    }
    if data[0..4] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid volume magic",
        ));
    }

    // Verify trailing CRC32 before parsing anything else.
    // CRC covers everything except the last 4 bytes (the CRC itself).
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes(
        data[data.len() - 4..]
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "truncated volume CRC"))?,
    );
    let computed_crc = crc32fast::hash(payload);
    if stored_crc != computed_crc {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "volume CRC mismatch: stored={:#x} computed={:#x}",
                stored_crc, computed_crc
            ),
        ));
    }

    pos += 4;
    let version = read_u32(data, &mut pos)?;
    if version != FORMAT_VERSION && version != FORMAT_VERSION_V2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported volume version {}", version),
        ));
    }
    let has_bloom_section = version >= FORMAT_VERSION_V2;
    let row_count = read_u64(data, &mut pos)? as usize;
    let col_count = read_u32(data, &mut pos)? as usize;
    pos += 12; // reserved

    // Column directory
    let mut col_types: Vec<u8> = Vec::with_capacity(col_count);
    let mut col_ext_types: Vec<u8> = Vec::with_capacity(col_count);
    let mut col_sorted: Vec<bool> = Vec::with_capacity(col_count);
    let mut col_dict_offsets: Vec<u32> = Vec::with_capacity(col_count);

    for _ in 0..col_count {
        if pos + 2 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated column directory",
            ));
        }
        let type_tag = data[pos];
        pos += 1;
        let flags = data[pos];
        pos += 1;
        let extra = read_u32(data, &mut pos)?;

        col_types.push(type_tag);
        // ext_type is in the 4-byte extra field for Bytes columns (not in flags)
        col_ext_types.push(if type_tag == COL_BYTES {
            extra as u8
        } else {
            0
        });
        col_sorted.push(flags & FLAG_SORTED != 0);
        col_dict_offsets.push(if type_tag == COL_DICTIONARY { extra } else { 0 });
    }

    // Column data
    let mut columns: Vec<ColumnData> = Vec::with_capacity(col_count);

    for i in 0..col_count {
        match col_types[i] {
            COL_INT64 => {
                let nulls = read_nulls(data, &mut pos, row_count)?;
                let values = read_i64_bulk(data, &mut pos, row_count)?;
                columns.push(ColumnData::Int64 { values, nulls });
            }
            COL_FLOAT64 => {
                let nulls = read_nulls(data, &mut pos, row_count)?;
                let values = read_f64_bulk(data, &mut pos, row_count)?;
                columns.push(ColumnData::Float64 { values, nulls });
            }
            COL_TIMESTAMP => {
                let nulls = read_nulls(data, &mut pos, row_count)?;
                let values = read_i64_bulk(data, &mut pos, row_count)?;
                columns.push(ColumnData::TimestampNanos { values, nulls });
            }
            COL_BOOLEAN => {
                let nulls = read_nulls(data, &mut pos, row_count)?;
                let values = read_bool_bulk(data, &mut pos, row_count)?;
                columns.push(ColumnData::Boolean { values, nulls });
            }
            COL_DICTIONARY => {
                let nulls = read_nulls(data, &mut pos, row_count)?;
                let ids = read_u32_bulk(data, &mut pos, row_count)?;
                // Dictionary will be filled after reading the shared dict
                columns.push(ColumnData::Dictionary {
                    ids,
                    dictionary: Vec::new(), // placeholder
                    nulls,
                });
            }
            COL_BYTES => {
                let nulls = read_nulls(data, &mut pos, row_count)?;
                let offset_count = read_u64(data, &mut pos)? as usize;
                let mut offsets = Vec::with_capacity(offset_count);
                for _ in 0..offset_count {
                    let off = read_u64(data, &mut pos)?;
                    let len = read_u64(data, &mut pos)?;
                    offsets.push((off, len));
                }
                let data_len = read_u64(data, &mut pos)? as usize;
                if pos + data_len > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "truncated volume: bytes column extends past end",
                    ));
                }
                let blob = data[pos..pos + data_len].to_vec();
                pos += data_len;
                columns.push(ColumnData::Bytes {
                    data: blob,
                    offsets,
                    ext_type: DataType::from_u8(col_ext_types[i]).unwrap_or(DataType::Null),
                    nulls,
                });
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown column type {}", col_types[i]),
                ));
            }
        }
    }

    // Shared string dictionary
    let dict_len = read_u32(data, &mut pos)? as usize;
    let mut shared_dict: Vec<SmartString> = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        let slen = read_u32(data, &mut pos)? as usize;
        if pos + slen > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated volume: dictionary string extends past end",
            ));
        }
        let s = std::str::from_utf8(&data[pos..pos + slen])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        shared_dict.push(SmartString::from(s));
        pos += slen;
    }

    // Fill in dictionaries for dictionary columns
    for (i, col) in columns.iter_mut().enumerate() {
        if col_types[i] == COL_DICTIONARY {
            if let ColumnData::Dictionary { dictionary, .. } = col {
                let start = col_dict_offsets[i] as usize;
                // Find end: next dict column's offset, or end of shared_dict
                let end = col_dict_offsets
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j > i && col_types[*j] == COL_DICTIONARY)
                    .map(|(_, &off)| off as usize)
                    .min()
                    .unwrap_or(shared_dict.len());
                *dictionary = shared_dict[start..end].to_vec();
            }
        }
    }

    // Row IDs
    let mut row_ids = Vec::with_capacity(row_count);
    for _ in 0..row_count {
        row_ids.push(read_i64(data, &mut pos)?);
    }

    // Zone maps
    let mut zone_maps = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        let min = read_value(data, &mut pos)?;
        let max = read_value(data, &mut pos)?;
        let null_count = read_u32(data, &mut pos)?;
        let row_count_zm = read_u32(data, &mut pos)?;
        zone_maps.push(ZoneMap {
            min,
            max,
            null_count,
            row_count: row_count_zm,
        });
    }

    // Bloom filters: read from disk (v2+) or defer rebuild for later (v1 backward compat)
    // Position: right after zone maps, matching the serialization order.
    let bloom_filters_raw: Option<Vec<super::column::ColumnBloomFilter>> = if has_bloom_section {
        let num_blooms = read_u32(data, &mut pos)? as usize;
        let mut filters = Vec::with_capacity(num_blooms);
        for _ in 0..num_blooms {
            let num_bits = read_u64(data, &mut pos)? as usize;
            let data_len = read_u32(data, &mut pos)? as usize;
            if pos + data_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated volume: bloom filter extends past end",
                ));
            }
            let bits_bytes = &data[pos..pos + data_len];
            pos += data_len;
            filters.push(super::column::ColumnBloomFilter::from_parts(
                num_bits, bits_bytes,
            ));
        }
        Some(filters)
    } else {
        None
    };

    // Stats
    let total_rows = read_u64(data, &mut pos)?;
    let live_rows = read_u64(data, &mut pos)?;
    let stats_col_count = read_u32(data, &mut pos)? as usize;
    let mut stat_columns = Vec::with_capacity(stats_col_count);
    for _ in 0..stats_col_count {
        let sum_int = read_i128(data, &mut pos)?;
        let sum_float = read_f64(data, &mut pos)?;
        let numeric_count = read_u64(data, &mut pos)?;
        let non_null_count = read_u64(data, &mut pos)?;
        let min = read_value(data, &mut pos)?;
        let max = read_value(data, &mut pos)?;
        stat_columns.push(ColumnAggregateStats {
            sum_int,
            sum_float,
            numeric_count,
            min,
            max,
            non_null_count,
        });
    }

    // Column names
    let mut column_names = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        let slen = read_u32(data, &mut pos)? as usize;
        if pos + slen > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated volume: column name extends past end",
            ));
        }
        let s = std::str::from_utf8(&data[pos..pos + slen])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        column_names.push(s.to_string());
        pos += slen;
    }

    // Column types
    let mut column_types = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        if pos >= data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated volume: missing column type data",
            ));
        }
        column_types.push(DataType::from_u8(data[pos]).unwrap_or(DataType::Null));
        pos += 1;
    }

    // Row groups (v3+): per-group zone maps for sub-volume pruning.
    // Read from file if present, otherwise compute from column data at load
    // time (same pattern as bloom filters for v1→v2). v2 files get row groups
    // computed once at startup without needing a format rewrite on disk.
    let row_groups_raw: Option<Vec<super::column::RowGroupMeta>> = if version >= FORMAT_VERSION {
        let num_groups = read_u32(data, &mut pos)? as usize;
        let mut groups = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            let start_idx = read_u32(data, &mut pos)?;
            let end_idx = read_u32(data, &mut pos)?;
            let mut group_zone_maps = Vec::with_capacity(col_count);
            for _ in 0..col_count {
                let min = read_value(data, &mut pos)?;
                let max = read_value(data, &mut pos)?;
                let nc = read_u32(data, &mut pos)?;
                let rc = read_u32(data, &mut pos)?;
                group_zone_maps.push(ZoneMap {
                    min,
                    max,
                    null_count: nc,
                    row_count: rc,
                });
            }
            groups.push(super::column::RowGroupMeta {
                start_idx,
                end_idx,
                zone_maps: group_zone_maps,
            });
        }
        Some(groups)
    } else {
        None
    };

    // Resolve bloom filters: use deserialized (v2+) or rebuild from column data
    let bloom_filters: Vec<super::column::ColumnBloomFilter> =
        bloom_filters_raw.unwrap_or_else(|| {
            columns
                .iter()
                .map(|col| {
                    let mut bf = super::column::ColumnBloomFilter::new(row_count.max(1));
                    for i in 0..row_count {
                        if !col.is_null(i) {
                            let value = col.get_value(i);
                            bf.add(&value);
                        }
                    }
                    bf
                })
                .collect()
        });

    let column_name_map: ahash::AHashMap<crate::common::SmartString, usize> = column_names
        .iter()
        .enumerate()
        .map(|(i, name)| (crate::common::SmartString::from(name.to_lowercase()), i))
        .collect();

    // Resolve row groups: use deserialized (v3+) or compute from column data.
    // Must happen before `columns` is moved into the struct.
    let row_groups = match row_groups_raw {
        Some(groups) => groups,
        None if row_count > super::column::ROW_GROUP_SIZE => {
            let mut groups = Vec::new();
            let mut start = 0;
            while start < row_count {
                let end = (start + super::column::ROW_GROUP_SIZE).min(row_count);
                let gzm: Vec<ZoneMap> = columns
                    .iter()
                    .map(|c| c.zone_map_for_range(start, end))
                    .collect();
                groups.push(super::column::RowGroupMeta {
                    start_idx: start as u32,
                    end_idx: end as u32,
                    zone_maps: gzm,
                });
                start = end;
            }
            groups
        }
        None => Vec::new(),
    };

    Ok(FrozenVolume {
        columns,
        zone_maps,
        bloom_filters,
        stats: VolumeAggregateStats {
            total_rows,
            live_rows,
            columns: stat_columns,
        },
        row_count,
        column_names,
        column_types,
        row_ids,
        sorted_columns: col_sorted,
        column_name_map,
        unique_indices: parking_lot::RwLock::new(rustc_hash::FxHashMap::default()),
        row_groups,
    })
}

// =============================================================================
// Helpers
// =============================================================================

fn write_nulls(buf: &mut Vec<u8>, nulls: &[bool]) -> io::Result<()> {
    for &n in nulls {
        buf.push(if n { 1 } else { 0 });
    }
    Ok(())
}

fn read_nulls(data: &[u8], pos: &mut usize, count: usize) -> io::Result<Vec<bool>> {
    let end = *pos + count;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: null bitmap extends past end of data",
        ));
    }
    let mut nulls = Vec::with_capacity(count);
    for i in 0..count {
        nulls.push(data[*pos + i] != 0);
    }
    *pos = end;
    Ok(nulls)
}

#[inline]
fn read_u32(data: &[u8], pos: &mut usize) -> io::Result<u32> {
    let end = *pos + 4;
    let bytes: [u8; 4] = data
        .get(*pos..end)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "truncated volume: expected u32")
        })?;
    *pos = end;
    Ok(u32::from_le_bytes(bytes))
}

#[inline]
fn read_u64(data: &[u8], pos: &mut usize) -> io::Result<u64> {
    let end = *pos + 8;
    let bytes: [u8; 8] = data
        .get(*pos..end)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "truncated volume: expected u64")
        })?;
    *pos = end;
    Ok(u64::from_le_bytes(bytes))
}

#[inline]
fn read_i64(data: &[u8], pos: &mut usize) -> io::Result<i64> {
    let end = *pos + 8;
    let bytes: [u8; 8] = data
        .get(*pos..end)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "truncated volume: expected i64")
        })?;
    *pos = end;
    Ok(i64::from_le_bytes(bytes))
}

#[inline]
fn read_f64(data: &[u8], pos: &mut usize) -> io::Result<f64> {
    let end = *pos + 8;
    let bytes: [u8; 8] = data
        .get(*pos..end)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "truncated volume: expected f64")
        })?;
    *pos = end;
    Ok(f64::from_le_bytes(bytes))
}

#[inline]
fn read_i128(data: &[u8], pos: &mut usize) -> io::Result<i128> {
    let end = *pos + 16;
    let bytes: [u8; 16] = data
        .get(*pos..end)
        .and_then(|s| s.try_into().ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated volume: expected i128",
            )
        })?;
    *pos = end;
    Ok(i128::from_le_bytes(bytes))
}

// =============================================================================
// Bulk read/write helpers for fixed-size columns.
// Single bounds check + memcpy instead of per-element function calls.
// =============================================================================

/// Write a slice of i64 values as little-endian bytes in bulk.
#[inline]
fn write_i64_bulk(buf: &mut Vec<u8>, values: &[i64]) {
    // On little-endian platforms, i64 in-memory layout matches the on-disk format.
    #[cfg(target_endian = "little")]
    {
        let byte_len = values.len() * 8;
        buf.reserve(byte_len);
        // SAFETY: &[i64] is layout-compatible with &[u8] on LE platforms.
        // The slice is valid for `byte_len` bytes (values.len() * size_of::<i64>()).
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) };
        buf.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        buf.reserve(values.len() * 8);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Write a slice of f64 values as little-endian bytes in bulk.
#[inline]
fn write_f64_bulk(buf: &mut Vec<u8>, values: &[f64]) {
    #[cfg(target_endian = "little")]
    {
        let byte_len = values.len() * 8;
        buf.reserve(byte_len);
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) };
        buf.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        buf.reserve(values.len() * 8);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Write a slice of u32 values as little-endian bytes in bulk.
#[inline]
fn write_u32_bulk(buf: &mut Vec<u8>, values: &[u32]) {
    #[cfg(target_endian = "little")]
    {
        let byte_len = values.len() * 4;
        buf.reserve(byte_len);
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, byte_len) };
        buf.extend_from_slice(bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        buf.reserve(values.len() * 4);
        for v in values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Write a slice of bool values as single bytes in bulk.
#[inline]
fn write_bool_bulk(buf: &mut Vec<u8>, values: &[bool]) {
    buf.reserve(values.len());
    for v in values {
        buf.push(if *v { 1 } else { 0 });
    }
}

/// Read `count` i64 values from little-endian bytes in bulk.
fn read_i64_bulk(data: &[u8], pos: &mut usize, count: usize) -> io::Result<Vec<i64>> {
    let byte_len = count * 8;
    let end = *pos + byte_len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: i64 column data",
        ));
    }
    #[cfg(target_endian = "little")]
    let values = {
        // On LE platforms, copy raw bytes directly into the i64 vec.
        let mut v = vec![0i64; count];
        // SAFETY: vec is initialized to zeros, copy_nonoverlapping overwrites all bytes.
        // Source bounds verified above. Layout of [i64] matches [u8; N*8] on LE.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[*pos..end].as_ptr(),
                v.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
        v
    };
    #[cfg(not(target_endian = "little"))]
    let values = {
        let mut v = Vec::with_capacity(count);
        for i in 0..count {
            let off = *pos + i * 8;
            let bytes: [u8; 8] = data[off..off + 8].try_into().unwrap();
            v.push(i64::from_le_bytes(bytes));
        }
        v
    };
    *pos = end;
    Ok(values)
}

/// Read `count` f64 values from little-endian bytes in bulk.
fn read_f64_bulk(data: &[u8], pos: &mut usize, count: usize) -> io::Result<Vec<f64>> {
    let byte_len = count * 8;
    let end = *pos + byte_len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: f64 column data",
        ));
    }
    #[cfg(target_endian = "little")]
    let values = {
        let mut v = vec![0f64; count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[*pos..end].as_ptr(),
                v.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
        v
    };
    #[cfg(not(target_endian = "little"))]
    let values = {
        let mut v = Vec::with_capacity(count);
        for i in 0..count {
            let off = *pos + i * 8;
            let bytes: [u8; 8] = data[off..off + 8].try_into().unwrap();
            v.push(f64::from_le_bytes(bytes));
        }
        v
    };
    *pos = end;
    Ok(values)
}

/// Read `count` u32 values from little-endian bytes in bulk.
fn read_u32_bulk(data: &[u8], pos: &mut usize, count: usize) -> io::Result<Vec<u32>> {
    let byte_len = count * 4;
    let end = *pos + byte_len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: u32 column data",
        ));
    }
    #[cfg(target_endian = "little")]
    let values = {
        let mut v = vec![0u32; count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[*pos..end].as_ptr(),
                v.as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
        v
    };
    #[cfg(not(target_endian = "little"))]
    let values = {
        let mut v = Vec::with_capacity(count);
        for i in 0..count {
            let off = *pos + i * 4;
            let bytes: [u8; 4] = data[off..off + 4].try_into().unwrap();
            v.push(u32::from_le_bytes(bytes));
        }
        v
    };
    *pos = end;
    Ok(values)
}

/// Read `count` boolean values from bytes in bulk.
fn read_bool_bulk(data: &[u8], pos: &mut usize, count: usize) -> io::Result<Vec<bool>> {
    let end = *pos + count;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: boolean column data",
        ));
    }
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        values.push(data[*pos + i] != 0);
    }
    *pos = end;
    Ok(values)
}

/// Serialize a Value to the buffer with a type tag.
fn write_value(buf: &mut Vec<u8>, value: &Value) -> io::Result<()> {
    match value {
        Value::Null(dt) => {
            buf.push(0);
            buf.push(*dt as u8);
        }
        Value::Integer(i) => {
            buf.push(1);
            buf.write_all(&i.to_le_bytes())?;
        }
        Value::Float(f) => {
            buf.push(2);
            buf.write_all(&f.to_le_bytes())?;
        }
        Value::Text(s) => {
            buf.push(3);
            let bytes = s.as_bytes();
            buf.write_all(&(bytes.len() as u32).to_le_bytes())?;
            buf.write_all(bytes)?;
        }
        Value::Boolean(b) => {
            buf.push(4);
            buf.push(if *b { 1 } else { 0 });
        }
        Value::Timestamp(ts) => {
            buf.push(5);
            let nanos = ts.timestamp_nanos_opt().unwrap_or_else(|| {
                ts.timestamp()
                    .wrapping_mul(1_000_000_000)
                    .wrapping_add(ts.timestamp_subsec_nanos() as i64)
            });
            buf.write_all(&nanos.to_le_bytes())?;
        }
        Value::Extension(data) => {
            buf.push(6);
            buf.write_all(&(data.len() as u32).to_le_bytes())?;
            buf.write_all(data)?;
        }
    }
    Ok(())
}

/// Deserialize a Value from the buffer.
fn read_value(data: &[u8], pos: &mut usize) -> io::Result<Value> {
    if *pos >= data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated value tag",
        ));
    }
    let tag = data[*pos];
    *pos += 1;
    match tag {
        0 => {
            if *pos >= data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated null type",
                ));
            }
            let dt = DataType::from_u8(data[*pos]).unwrap_or(DataType::Null);
            *pos += 1;
            Ok(Value::Null(dt))
        }
        1 => Ok(Value::Integer(read_i64(data, pos)?)),
        2 => Ok(Value::Float(read_f64(data, pos)?)),
        3 => {
            let slen = read_u32(data, pos)? as usize;
            if *pos + slen > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated volume: text value data",
                ));
            }
            let s = std::str::from_utf8(&data[*pos..*pos + slen])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            *pos += slen;
            Ok(Value::text(s))
        }
        4 => {
            if *pos >= data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated boolean",
                ));
            }
            let b = data[*pos] != 0;
            *pos += 1;
            Ok(Value::Boolean(b))
        }
        5 => {
            let nanos = read_i64(data, pos)?;
            let secs = nanos.div_euclid(1_000_000_000);
            let sub_nanos = nanos.rem_euclid(1_000_000_000) as u32;
            match chrono::TimeZone::timestamp_opt(&chrono::Utc, secs, sub_nanos) {
                chrono::LocalResult::Single(dt) => Ok(Value::Timestamp(dt)),
                _ => Ok(Value::Null(DataType::Timestamp)),
            }
        }
        6 => {
            let len = read_u32(data, pos)? as usize;
            if *pos + len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated extension data",
                ));
            }
            let bytes = data[*pos..*pos + len].to_vec();
            *pos += len;
            Ok(Value::Extension(crate::common::CompactArc::from(bytes)))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown value tag {}", tag),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::super::writer::VolumeBuilder;
    use super::*;
    use crate::core::{Row, SchemaBuilder};

    #[test]
    fn test_roundtrip_basic() {
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
        let original = builder.finish();

        // Serialize
        let bytes = serialize_volume(&original).unwrap();
        assert!(bytes.len() > 32); // at least header

        // Deserialize
        let loaded = deserialize_volume(&bytes).unwrap();

        // Verify
        assert_eq!(loaded.row_count, 3);
        assert_eq!(loaded.columns.len(), 3);
        assert_eq!(loaded.column_names, vec!["id", "name", "price"]);
        assert_eq!(loaded.row_ids, vec![1, 2, 3]);

        // Check values
        assert_eq!(loaded.columns[0].get_i64(0), 1);
        assert_eq!(loaded.columns[0].get_i64(2), 3);
        assert_eq!(loaded.columns[1].get_str(0), "apple");
        assert_eq!(loaded.columns[1].get_str(1), "banana");
        assert_eq!(loaded.columns[2].get_f64(1), 0.75);

        // Check zone maps survived
        assert_eq!(loaded.zone_maps[0].min, Value::Integer(1));
        assert_eq!(loaded.zone_maps[0].max, Value::Integer(3));

        // Check stats survived
        assert_eq!(loaded.stats.count_star(), 3);
        assert_eq!(loaded.stats.sum(2), 5.25); // 1.50 + 0.75 + 3.00

        // Check sorted flags
        assert!(loaded.sorted_columns[0]); // id is sorted
    }

    #[test]
    fn test_roundtrip_with_nulls() {
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
        let original = builder.finish();

        let bytes = serialize_volume(&original).unwrap();
        let loaded = deserialize_volume(&bytes).unwrap();

        assert!(!loaded.columns[1].is_null(0));
        assert!(loaded.columns[1].is_null(1));
        assert!(!loaded.columns[1].is_null(2));
        assert_eq!(loaded.columns[1].get_f64(0), 10.0);
        assert_eq!(loaded.columns[1].get_f64(2), 30.0);
    }

    #[test]
    fn test_roundtrip_timestamp() {
        let schema = SchemaBuilder::new("test")
            .column("time", DataType::Timestamp, false, false)
            .build();

        let ts = chrono::Utc::now();
        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Timestamp(ts)]));
        let original = builder.finish();

        let bytes = serialize_volume(&original).unwrap();
        let loaded = deserialize_volume(&bytes).unwrap();

        // Timestamps preserve nanosecond precision
        let loaded_val = loaded.columns[0].get_value(0);
        if let (Value::Timestamp(orig), Value::Timestamp(load)) =
            (&Value::Timestamp(ts), &loaded_val)
        {
            assert_eq!(
                orig.timestamp_nanos_opt(),
                load.timestamp_nanos_opt(),
                "timestamp nanoseconds must match"
            );
        } else {
            panic!("expected timestamps");
        }
    }

    #[test]
    fn test_roundtrip_boolean() {
        let schema = SchemaBuilder::new("test")
            .column("flag", DataType::Boolean, false, false)
            .build();

        let mut builder = VolumeBuilder::new(&schema);
        builder.add_row(1, &Row::from_values(vec![Value::Boolean(true)]));
        builder.add_row(2, &Row::from_values(vec![Value::Boolean(false)]));
        builder.add_row(3, &Row::from_values(vec![Value::Boolean(true)]));
        let original = builder.finish();

        let bytes = serialize_volume(&original).unwrap();
        let loaded = deserialize_volume(&bytes).unwrap();

        assert!(loaded.columns[0].get_bool(0));
        assert!(!loaded.columns[0].get_bool(1));
        assert!(loaded.columns[0].get_bool(2));
    }

    #[test]
    fn test_file_size_vs_snapshot() {
        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .column("exchange", DataType::Text, false, false)
            .column("symbol", DataType::Text, false, false)
            .column("price", DataType::Float, false, false)
            .build();

        let exchanges = ["binance", "coinbase"];
        let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];

        let mut builder = VolumeBuilder::with_capacity(&schema, 10000);
        for i in 0..10000 {
            builder.add_row(
                i,
                &Row::from_values(vec![
                    Value::Integer(i),
                    Value::text(exchanges[i as usize % 2]),
                    Value::text(symbols[i as usize % 3]),
                    Value::Float(100.0 + i as f64 * 0.01),
                ]),
            );
        }
        let vol = builder.finish();
        let bytes = serialize_volume(&vol).unwrap();

        // Column-major + dictionary encoding should be much smaller
        // than row-major snapshot format (which repeats exchange/symbol strings)
        let row_major_estimate = 10000 * (9 + 8 + 8 + 8 + 8); // rough: tags + values
        eprintln!(
            "Volume: {} bytes, row-major estimate: {} bytes, ratio: {:.1}%",
            bytes.len(),
            row_major_estimate,
            bytes.len() as f64 / row_major_estimate as f64 * 100.0
        );

        // Roundtrip
        let loaded = deserialize_volume(&bytes).unwrap();
        assert_eq!(loaded.row_count, 10000);
        assert_eq!(loaded.columns[1].get_str(0), "binance");
        assert_eq!(loaded.columns[1].get_str(1), "coinbase");
    }
}
