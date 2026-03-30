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
use std::sync::Arc;

use crate::common::SmartString;
use crate::core::{DataType, Value};

use super::column::{ColumnData, ZoneMap};
use super::stats::{ColumnAggregateStats, VolumeAggregateStats};
use super::writer::FrozenVolume;

// Column type tags for the directory
pub(crate) const COL_INT64: u8 = 1;
pub(crate) const COL_FLOAT64: u8 = 2;
pub(crate) const COL_TIMESTAMP: u8 = 3;
pub(crate) const COL_BOOLEAN: u8 = 4;
pub(crate) const COL_DICTIONARY: u8 = 5;
pub(crate) const COL_BYTES: u8 = 6;

// Column flags
pub(crate) const FLAG_SORTED: u8 = 0x01;

// STVL format removed. All volumes use V4 format (see io.rs).

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

// =============================================================================
// V4 format: per-row-group per-column block serialization
// =============================================================================

/// Serialize a single column's row range [start, end) to raw bytes.
/// For Dictionary columns, only nulls+ids are written (dictionary stored separately).
pub(crate) fn serialize_column_block(col: &ColumnData, start: usize, end: usize) -> Vec<u8> {
    let count = end - start;
    let estimated = match col {
        ColumnData::Int64 { .. } | ColumnData::TimestampNanos { .. } => count * 9,
        ColumnData::Float64 { .. } => count * 9,
        ColumnData::Boolean { .. } => count * 2,
        ColumnData::Dictionary { .. } => count * 5,
        ColumnData::Bytes { .. } => count * 17,
    };
    let mut buf = Vec::with_capacity(estimated);
    match col {
        ColumnData::Int64 { values, nulls } => {
            write_nulls(&mut buf, &nulls[start..end]).unwrap();
            write_i64_bulk(&mut buf, &values[start..end]);
        }
        ColumnData::Float64 { values, nulls } => {
            write_nulls(&mut buf, &nulls[start..end]).unwrap();
            write_f64_bulk(&mut buf, &values[start..end]);
        }
        ColumnData::TimestampNanos { values, nulls } => {
            write_nulls(&mut buf, &nulls[start..end]).unwrap();
            write_i64_bulk(&mut buf, &values[start..end]);
        }
        ColumnData::Boolean { values, nulls } => {
            write_nulls(&mut buf, &nulls[start..end]).unwrap();
            write_bool_bulk(&mut buf, &values[start..end]);
        }
        ColumnData::Dictionary { ids, nulls, .. } => {
            // Dictionary stored separately — only write nulls + ids
            write_nulls(&mut buf, &nulls[start..end]).unwrap();
            write_u32_bulk(&mut buf, &ids[start..end]);
        }
        ColumnData::Bytes {
            data,
            offsets,
            nulls,
            ..
        } => {
            write_nulls(&mut buf, &nulls[start..end]).unwrap();
            let range_offsets = &offsets[start..end];
            let count = end - start;
            // Repack data blob with zero-based offsets for this block
            let mut new_data = Vec::new();
            let mut new_offsets = Vec::with_capacity(count);
            for &(off, len) in range_offsets {
                let new_off = new_data.len() as u64;
                if len > 0 && (off as usize) < data.len() {
                    let end_pos = ((off + len) as usize).min(data.len());
                    new_data.extend_from_slice(&data[off as usize..end_pos]);
                }
                new_offsets.push((new_off, len));
            }
            buf.write_all(&(new_offsets.len() as u64).to_le_bytes())
                .unwrap();
            for (off, len) in &new_offsets {
                buf.write_all(&off.to_le_bytes()).unwrap();
                buf.write_all(&len.to_le_bytes()).unwrap();
            }
            buf.write_all(&(new_data.len() as u64).to_le_bytes())
                .unwrap();
            buf.write_all(&new_data).unwrap();
        }
    }
    buf
}

// =============================================================================
// Append-into helpers: extend existing Vecs instead of allocating new ones.
// Used by CompressedBlockStore::decompress_column for multi-group columns to
// avoid 1000s of intermediate ColumnData allocations.
// =============================================================================

/// Append `count` null flags from `data[*pos..]` into `out`.
#[inline]
pub(crate) fn read_nulls_into(
    data: &[u8],
    pos: &mut usize,
    count: usize,
    out: &mut Vec<bool>,
) -> io::Result<()> {
    let end = *pos + count;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: null bitmap extends past end of data",
        ));
    }
    out.reserve(count);
    for i in 0..count {
        out.push(data[*pos + i] != 0);
    }
    *pos = end;
    Ok(())
}

/// Append `count` i64 values from `data[*pos..]` into `out` (bulk, LE).
#[inline]
pub(crate) fn read_i64_bulk_into(
    data: &[u8],
    pos: &mut usize,
    count: usize,
    out: &mut Vec<i64>,
) -> io::Result<()> {
    let byte_len = count * 8;
    let end = *pos + byte_len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: i64 column data",
        ));
    }
    #[cfg(target_endian = "little")]
    {
        let old_len = out.len();
        out.resize(old_len + count, 0i64);
        // SAFETY: slice is valid for byte_len bytes; layout of [i64] matches [u8; N*8] on LE.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[*pos..end].as_ptr(),
                out[old_len..].as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        out.reserve(count);
        for i in 0..count {
            let off = *pos + i * 8;
            let bytes: [u8; 8] = data[off..off + 8].try_into().unwrap();
            out.push(i64::from_le_bytes(bytes));
        }
    }
    *pos = end;
    Ok(())
}

/// Append `count` f64 values from `data[*pos..]` into `out` (bulk, LE).
#[inline]
pub(crate) fn read_f64_bulk_into(
    data: &[u8],
    pos: &mut usize,
    count: usize,
    out: &mut Vec<f64>,
) -> io::Result<()> {
    let byte_len = count * 8;
    let end = *pos + byte_len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: f64 column data",
        ));
    }
    #[cfg(target_endian = "little")]
    {
        let old_len = out.len();
        out.resize(old_len + count, 0f64);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[*pos..end].as_ptr(),
                out[old_len..].as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        out.reserve(count);
        for i in 0..count {
            let off = *pos + i * 8;
            let bytes: [u8; 8] = data[off..off + 8].try_into().unwrap();
            out.push(f64::from_le_bytes(bytes));
        }
    }
    *pos = end;
    Ok(())
}

/// Append `count` u32 values from `data[*pos..]` into `out` (bulk, LE).
#[inline]
pub(crate) fn read_u32_bulk_into(
    data: &[u8],
    pos: &mut usize,
    count: usize,
    out: &mut Vec<u32>,
) -> io::Result<()> {
    let byte_len = count * 4;
    let end = *pos + byte_len;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: u32 column data",
        ));
    }
    #[cfg(target_endian = "little")]
    {
        let old_len = out.len();
        out.resize(old_len + count, 0u32);
        unsafe {
            std::ptr::copy_nonoverlapping(
                data[*pos..end].as_ptr(),
                out[old_len..].as_mut_ptr() as *mut u8,
                byte_len,
            );
        }
    }
    #[cfg(not(target_endian = "little"))]
    {
        out.reserve(count);
        for i in 0..count {
            let off = *pos + i * 4;
            let bytes: [u8; 4] = data[off..off + 4].try_into().unwrap();
            out.push(u32::from_le_bytes(bytes));
        }
    }
    *pos = end;
    Ok(())
}

/// Append `count` bool values from `data[*pos..]` into `out`.
#[inline]
pub(crate) fn read_bool_bulk_into(
    data: &[u8],
    pos: &mut usize,
    count: usize,
    out: &mut Vec<bool>,
) -> io::Result<()> {
    let end = *pos + count;
    if end > data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: boolean column data",
        ));
    }
    out.reserve(count);
    for i in 0..count {
        out.push(data[*pos + i] != 0);
    }
    *pos = end;
    Ok(())
}

/// Deserialize a single column block, appending directly into the caller's
/// output buffers. This avoids creating an intermediate `ColumnData` per group.
///
/// - `values_out`: receives the column's typed values (i64/f64/u32/bool).
///   For COL_BYTES the bytes blob is returned separately; pass `()` as
///   the generic: the function writes data/offsets into the dedicated parameters.
/// - `nulls_out`: receives the null flags.
///
/// For COL_BYTES, `bytes_data_out` and `bytes_offsets_out` are used instead of
/// `values_out`. The offset base for the current group must be passed as
/// `bytes_base_offset` so block-local offsets are converted to global offsets.
#[allow(clippy::too_many_arguments)]
pub(crate) fn deserialize_column_block_into(
    data: &[u8],
    col_type_tag: u8,
    row_count: usize,
    nulls_out: &mut Vec<bool>,
    // Typed value outputs — only the one matching col_type_tag is used:
    i64_out: Option<&mut Vec<i64>>,
    f64_out: Option<&mut Vec<f64>>,
    u32_out: Option<&mut Vec<u32>>,
    bool_out: Option<&mut Vec<bool>>,
    // COL_BYTES specific:
    bytes_data_out: Option<&mut Vec<u8>>,
    bytes_offsets_out: Option<&mut Vec<(u64, u64)>>,
) -> io::Result<()> {
    let mut pos = 0;
    match col_type_tag {
        COL_INT64 | COL_TIMESTAMP => {
            read_nulls_into(data, &mut pos, row_count, nulls_out)?;
            read_i64_bulk_into(
                data,
                &mut pos,
                row_count,
                i64_out.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing i64_out buffer")
                })?,
            )?;
        }
        COL_FLOAT64 => {
            read_nulls_into(data, &mut pos, row_count, nulls_out)?;
            read_f64_bulk_into(
                data,
                &mut pos,
                row_count,
                f64_out.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing f64_out buffer")
                })?,
            )?;
        }
        COL_BOOLEAN => {
            read_nulls_into(data, &mut pos, row_count, nulls_out)?;
            read_bool_bulk_into(
                data,
                &mut pos,
                row_count,
                bool_out.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing bool_out buffer")
                })?,
            )?;
        }
        COL_DICTIONARY => {
            read_nulls_into(data, &mut pos, row_count, nulls_out)?;
            read_u32_bulk_into(
                data,
                &mut pos,
                row_count,
                u32_out.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "missing u32_out buffer")
                })?,
            )?;
        }
        COL_BYTES => {
            read_nulls_into(data, &mut pos, row_count, nulls_out)?;
            let offset_count = read_u64(data, &mut pos)? as usize;
            let bytes_data = bytes_data_out.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing bytes_data_out buffer")
            })?;
            let bytes_offsets = bytes_offsets_out.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "missing bytes_offsets_out buffer",
                )
            })?;
            let base = bytes_data.len() as u64;
            bytes_offsets.reserve(offset_count);
            for _ in 0..offset_count {
                let off = read_u64(data, &mut pos)?;
                let len = read_u64(data, &mut pos)?;
                bytes_offsets.push((off + base, len));
            }
            let data_len = read_u64(data, &mut pos)? as usize;
            if pos + data_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated column block: bytes data",
                ));
            }
            bytes_data.extend_from_slice(&data[pos..pos + data_len]);
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown column type tag {}", col_type_tag),
            ));
        }
    }
    Ok(())
}

/// Deserialize a single column block from raw bytes.
/// For Dictionary columns, pass the dictionary; for Bytes columns, pass ext_type.
pub(crate) fn deserialize_column_block(
    data: &[u8],
    col_type_tag: u8,
    row_count: usize,
    dictionary: Option<Arc<[SmartString]>>,
    ext_type: DataType,
) -> io::Result<ColumnData> {
    let mut pos = 0;
    match col_type_tag {
        COL_INT64 => {
            let nulls = read_nulls(data, &mut pos, row_count)?;
            let values = read_i64_bulk(data, &mut pos, row_count)?;
            Ok(ColumnData::Int64 { values, nulls })
        }
        COL_FLOAT64 => {
            let nulls = read_nulls(data, &mut pos, row_count)?;
            let values = read_f64_bulk(data, &mut pos, row_count)?;
            Ok(ColumnData::Float64 { values, nulls })
        }
        COL_TIMESTAMP => {
            let nulls = read_nulls(data, &mut pos, row_count)?;
            let values = read_i64_bulk(data, &mut pos, row_count)?;
            Ok(ColumnData::TimestampNanos { values, nulls })
        }
        COL_BOOLEAN => {
            let nulls = read_nulls(data, &mut pos, row_count)?;
            let values = read_bool_bulk(data, &mut pos, row_count)?;
            Ok(ColumnData::Boolean { values, nulls })
        }
        COL_DICTIONARY => {
            let nulls = read_nulls(data, &mut pos, row_count)?;
            let ids = read_u32_bulk(data, &mut pos, row_count)?;
            Ok(ColumnData::Dictionary {
                ids,
                dictionary: dictionary.unwrap_or_else(|| Arc::from(Vec::<SmartString>::new())),
                nulls,
            })
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
                    "truncated column block: bytes data",
                ));
            }
            let blob = data[pos..pos + data_len].to_vec();
            Ok(ColumnData::Bytes {
                data: blob,
                offsets,
                ext_type,
                nulls,
            })
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown column type tag {}", col_type_tag),
        )),
    }
}

/// Metadata parsed from a V4 volume file (everything except column data).
pub(crate) struct VolumeMetadata {
    pub row_count: usize,
    #[allow(dead_code)]
    pub col_count: usize,
    pub col_type_tags: Vec<u8>,
    pub col_ext_types: Vec<u8>,
    pub col_sorted: Vec<bool>,
    pub col_dict_counts: Vec<u32>,
    pub shared_dict: Vec<SmartString>,
    pub row_ids: Vec<i64>,
    pub zone_maps: Vec<ZoneMap>,
    pub bloom_filters: Vec<super::column::ColumnBloomFilter>,
    pub stats: VolumeAggregateStats,
    pub column_names: Vec<String>,
    pub column_types: Vec<DataType>,
    pub row_groups: Vec<super::column::RowGroupMeta>,
    pub column_name_map: ahash::AHashMap<SmartString, usize>,
}

/// Serialize volume metadata (everything except column data) for V4 format.
/// The caller LZ4-compresses the result before writing to disk.
pub(crate) fn serialize_volume_metadata(vol: &FrozenVolume) -> io::Result<Vec<u8>> {
    let col_count = vol.columns.len();
    let estimated = 12 + col_count * 6 + vol.meta.row_ids.len() * 8 + col_count * 40;
    let mut buf = Vec::with_capacity(estimated);

    // Row count + col count
    buf.write_all(&(vol.meta.row_count as u64).to_le_bytes())?;
    buf.write_all(&(col_count as u32).to_le_bytes())?;

    // Build shared dict from Dictionary columns
    let mut shared_dict: Vec<SmartString> = Vec::new();
    let mut dict_counts: Vec<u32> = Vec::new();
    for i in 0..col_count {
        if let ColumnData::Dictionary { dictionary, .. } = &vol.columns[i] {
            dict_counts.push(dictionary.len() as u32);
            shared_dict.extend(dictionary.iter().cloned());
        }
    }

    // Column directory: type(1) + flags(1) + extra(4) per column
    let mut dict_col_idx = 0usize;
    for i in 0..col_count {
        let col = &vol.columns[i];
        let type_tag = match col {
            ColumnData::Int64 { .. } => COL_INT64,
            ColumnData::Float64 { .. } => COL_FLOAT64,
            ColumnData::TimestampNanos { .. } => COL_TIMESTAMP,
            ColumnData::Boolean { .. } => COL_BOOLEAN,
            ColumnData::Dictionary { .. } => COL_DICTIONARY,
            ColumnData::Bytes { .. } => COL_BYTES,
        };
        let sorted_flag = if vol.meta.sorted_columns[i] {
            FLAG_SORTED
        } else {
            0
        };
        buf.push(type_tag);
        buf.push(sorted_flag);
        if type_tag == COL_DICTIONARY {
            buf.write_all(&dict_counts[dict_col_idx].to_le_bytes())?;
            dict_col_idx += 1;
        } else if type_tag == COL_BYTES {
            let ext = match col {
                ColumnData::Bytes { ext_type, .. } => *ext_type as u32,
                _ => 0,
            };
            buf.write_all(&ext.to_le_bytes())?;
        } else {
            buf.write_all(&[0u8; 4])?;
        }
    }

    // Shared dictionary
    buf.write_all(&(shared_dict.len() as u32).to_le_bytes())?;
    for s in &shared_dict {
        let bytes = s.as_bytes();
        buf.write_all(&(bytes.len() as u32).to_le_bytes())?;
        buf.write_all(bytes)?;
    }

    // Row IDs (bulk — single memcpy on LE)
    write_i64_bulk(&mut buf, &vol.meta.row_ids);

    // Zone maps
    for zm in &vol.meta.zone_maps {
        write_value(&mut buf, &zm.min)?;
        write_value(&mut buf, &zm.max)?;
        buf.write_all(&zm.null_count.to_le_bytes())?;
        buf.write_all(&zm.row_count.to_le_bytes())?;
    }

    // Bloom filters
    buf.write_all(&(vol.meta.bloom_filters.len() as u32).to_le_bytes())?;
    for bf in &vol.meta.bloom_filters {
        buf.write_all(&(bf.num_bits() as u64).to_le_bytes())?;
        let data_bytes = bf.bits_as_bytes();
        buf.write_all(&(data_bytes.len() as u32).to_le_bytes())?;
        buf.write_all(&data_bytes)?;
    }

    // Stats
    buf.write_all(&vol.meta.stats.total_rows.to_le_bytes())?;
    buf.write_all(&vol.meta.stats.live_rows.to_le_bytes())?;
    buf.write_all(&(vol.meta.stats.columns.len() as u32).to_le_bytes())?;
    for cs in &vol.meta.stats.columns {
        buf.write_all(&cs.sum_int.to_le_bytes())?;
        buf.write_all(&cs.sum_float.to_le_bytes())?;
        buf.write_all(&cs.numeric_count.to_le_bytes())?;
        buf.write_all(&cs.non_null_count.to_le_bytes())?;
        write_value(&mut buf, &cs.min)?;
        write_value(&mut buf, &cs.max)?;
    }

    // Column names
    for name in &vol.meta.column_names {
        let bytes = name.as_bytes();
        buf.write_all(&(bytes.len() as u32).to_le_bytes())?;
        buf.write_all(bytes)?;
    }

    // Column types
    for dt in &vol.meta.column_types {
        buf.push(*dt as u8);
    }

    // Row groups
    buf.write_all(&(vol.meta.row_groups.len() as u32).to_le_bytes())?;
    for rg in &vol.meta.row_groups {
        buf.write_all(&rg.start_idx.to_le_bytes())?;
        buf.write_all(&rg.end_idx.to_le_bytes())?;
        for zm in &rg.zone_maps {
            write_value(&mut buf, &zm.min)?;
            write_value(&mut buf, &zm.max)?;
            buf.write_all(&zm.null_count.to_le_bytes())?;
            buf.write_all(&zm.row_count.to_le_bytes())?;
        }
    }

    Ok(buf)
}

/// Deserialize volume metadata from V4 format bytes.
pub(crate) fn deserialize_volume_metadata(data: &[u8]) -> io::Result<VolumeMetadata> {
    let mut pos = 0;

    let row_count = read_u64(data, &mut pos)? as usize;
    let col_count = read_u32(data, &mut pos)? as usize;

    // Column directory
    let mut col_type_tags = Vec::with_capacity(col_count);
    let mut col_ext_types = Vec::with_capacity(col_count);
    let mut col_sorted = Vec::with_capacity(col_count);
    let mut col_dict_counts = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        if pos + 2 > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated V4 column directory",
            ));
        }
        let type_tag = data[pos];
        pos += 1;
        let flags = data[pos];
        pos += 1;
        let extra = read_u32(data, &mut pos)?;
        col_type_tags.push(type_tag);
        col_ext_types.push(if type_tag == COL_BYTES {
            extra as u8
        } else {
            0
        });
        col_sorted.push(flags & FLAG_SORTED != 0);
        col_dict_counts.push(if type_tag == COL_DICTIONARY { extra } else { 0 });
    }

    // Shared dictionary
    let dict_len = read_u32(data, &mut pos)? as usize;
    let mut shared_dict = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        let slen = read_u32(data, &mut pos)? as usize;
        if pos + slen > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated V4 metadata: dictionary string",
            ));
        }
        let s = std::str::from_utf8(&data[pos..pos + slen])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        shared_dict.push(SmartString::from(s));
        pos += slen;
    }

    // Row IDs (bulk read — single memcpy on LE platforms)
    let row_ids = read_i64_bulk(data, &mut pos, row_count)?;

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

    // Bloom filters
    let num_blooms = read_u32(data, &mut pos)? as usize;
    let mut bloom_filters = Vec::with_capacity(num_blooms);
    for _ in 0..num_blooms {
        let num_bits = read_u64(data, &mut pos)? as usize;
        let data_len = read_u32(data, &mut pos)? as usize;
        if pos + data_len > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated V4 metadata: bloom filter",
            ));
        }
        let bits_bytes = &data[pos..pos + data_len];
        pos += data_len;
        bloom_filters.push(super::column::ColumnBloomFilter::from_parts(
            num_bits, bits_bytes,
        ));
    }

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
                "truncated V4 metadata: column name",
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
                "truncated V4 metadata: column type",
            ));
        }
        column_types.push(DataType::from_u8(data[pos]).unwrap_or(DataType::Null));
        pos += 1;
    }

    // Row groups
    let num_groups = read_u32(data, &mut pos)? as usize;
    let mut row_groups = Vec::with_capacity(num_groups);
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
        row_groups.push(super::column::RowGroupMeta {
            start_idx,
            end_idx,
            zone_maps: group_zone_maps,
        });
    }

    let column_name_map = column_names
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
        .collect();

    Ok(VolumeMetadata {
        row_count,
        col_count,
        col_type_tags,
        col_ext_types,
        col_sorted,
        col_dict_counts,
        shared_dict,
        row_ids,
        zone_maps,
        bloom_filters,
        stats: VolumeAggregateStats {
            total_rows,
            live_rows,
            columns: stat_columns,
        },
        column_names,
        column_types,
        row_groups,
        column_name_map,
    })
}
