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

use std::borrow::Cow;
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
    let Some(bytes) = data.get(*pos..end) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: null bitmap extends past end of data",
        ));
    };
    let nulls = bytes.iter().map(|&b| b != 0).collect();
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

/// Write i64 values as little-endian bytes to any writer, the slice's
/// own bytes on a little-endian platform
fn write_i64_le<W: Write>(w: &mut W, values: &[i64]) -> io::Result<()> {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: &[i64] is layout-compatible with &[u8] on LE platforms.
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8) };
        w.write_all(bytes)
    }
    #[cfg(not(target_endian = "little"))]
    {
        for v in values {
            w.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
}

/// Write u64 values as little-endian bytes to any writer
fn write_u64_le<W: Write>(w: &mut W, values: &[u64]) -> io::Result<()> {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: &[u64] is layout-compatible with &[u8] on LE platforms.
        let bytes =
            unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 8) };
        w.write_all(bytes)
    }
    #[cfg(not(target_endian = "little"))]
    {
        for v in values {
            w.write_all(&v.to_le_bytes())?;
        }
        Ok(())
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
fn write_value<W: Write>(buf: &mut W, value: &Value) -> io::Result<()> {
    match value {
        Value::Null(dt) => {
            buf.write_all(&[0])?;
            buf.write_all(&[*dt as u8])?;
        }
        Value::Integer(i) => {
            buf.write_all(&[1])?;
            buf.write_all(&i.to_le_bytes())?;
        }
        Value::Float(f) => {
            buf.write_all(&[2])?;
            buf.write_all(&f.to_le_bytes())?;
        }
        Value::Text(s) => {
            buf.write_all(&[3])?;
            let bytes = s.as_bytes();
            buf.write_all(&(bytes.len() as u32).to_le_bytes())?;
            buf.write_all(bytes)?;
        }
        Value::Boolean(b) => {
            buf.write_all(&[4])?;
            buf.write_all(&[if *b { 1 } else { 0 }])?;
        }
        Value::Timestamp(ts) => {
            buf.write_all(&[5])?;
            let nanos = ts.timestamp_nanos_opt().unwrap_or_else(|| {
                ts.timestamp()
                    .wrapping_mul(1_000_000_000)
                    .wrapping_add(ts.timestamp_subsec_nanos() as i64)
            });
            buf.write_all(&nanos.to_le_bytes())?;
        }
        Value::Extension(data) => {
            buf.write_all(&[6])?;
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
            if slen > data.len() - *pos {
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
            if len > data.len() - *pos {
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
/// One block from typed cells, the encoding `serialize_column_block` gives
/// the same cells inside a column
pub(crate) fn serialize_typed_block(cells: &super::writer::TypedCells<'_>) -> Vec<u8> {
    use super::writer::TypedCells;
    let count = cells.len();
    let mut buf = Vec::with_capacity(match cells {
        TypedCells::Int64 { .. } | TypedCells::TimestampNanos { .. } => count * 9,
        TypedCells::Float64 { .. } => count * 9,
        TypedCells::Boolean { .. } => count * 2,
        TypedCells::Dictionary { .. } => count * 5,
        TypedCells::Bytes { .. } => count * 17,
    });
    match cells {
        TypedCells::Int64 { values, nulls } | TypedCells::TimestampNanos { values, nulls } => {
            write_nulls(&mut buf, nulls).unwrap();
            write_i64_bulk(&mut buf, values);
        }
        TypedCells::Float64 { values, nulls } => {
            write_nulls(&mut buf, nulls).unwrap();
            write_f64_bulk(&mut buf, values);
        }
        TypedCells::Boolean { values, nulls } => {
            write_nulls(&mut buf, nulls).unwrap();
            write_bool_bulk(&mut buf, values);
        }
        TypedCells::Dictionary { ids, nulls } => {
            write_nulls(&mut buf, nulls).unwrap();
            write_u32_bulk(&mut buf, ids);
        }
        TypedCells::Bytes {
            data,
            offsets,
            nulls,
        } => {
            write_nulls(&mut buf, nulls).unwrap();
            let mut new_data = Vec::new();
            let mut new_offsets = Vec::with_capacity(count);
            for &(off, len) in offsets.iter() {
                let new_off = new_data.len() as u64;
                if len > 0 && (off as usize) < data.len() {
                    let end_pos = ((off + len) as usize).min(data.len());
                    let actual_len = (end_pos - off as usize) as u64;
                    new_data.extend_from_slice(&data[off as usize..end_pos]);
                    new_offsets.push((new_off, actual_len));
                } else {
                    new_offsets.push((new_off, 0));
                }
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
                    let actual_len = (end_pos - off as usize) as u64;
                    new_data.extend_from_slice(&data[off as usize..end_pos]);
                    new_offsets.push((new_off, actual_len));
                } else {
                    new_offsets.push((new_off, 0));
                }
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
    let Some(bytes) = data.get(*pos..end) else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated volume: null bitmap extends past end of data",
        ));
    };
    out.extend(bytes.iter().map(|&b| b != 0));
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
            let offset_count = read_u64(data, &mut pos)?;
            let bytes_data = bytes_data_out.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "missing bytes_data_out buffer")
            })?;
            let bytes_offsets = bytes_offsets_out.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "missing bytes_offsets_out buffer",
                )
            })?;
            if offset_count != row_count as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "bytes offset_count {} != row_count {}",
                        offset_count, row_count
                    ),
                ));
            }
            if row_count > data.len().saturating_sub(pos) / 16 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated column block: bytes offsets",
                ));
            }
            let base = bytes_data.len() as u64;
            bytes_offsets.reserve(row_count);
            for _ in 0..row_count {
                let off = read_u64(data, &mut pos)?;
                let len = read_u64(data, &mut pos)?;
                let adjusted = off.checked_add(base).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "bytes offset overflow")
                })?;
                bytes_offsets.push((adjusted, len));
            }
            let data_len = read_u64(data, &mut pos)?;
            if data_len > data.len().saturating_sub(pos) as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated column block: bytes data",
                ));
            }
            let end_pos = pos + data_len as usize;
            bytes_data.extend_from_slice(&data[pos..end_pos]);
            pos = end_pos;
            // Validate offsets against the data blob
            for (i, &(off, len)) in bytes_offsets
                .iter()
                .skip(bytes_offsets.len() - row_count)
                .enumerate()
            {
                let end = off.checked_add(len).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("bytes offset overflow at row {}", i),
                    )
                })?;
                if end > bytes_data.len() as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "bytes offset {}+{} exceeds data length {} at row {}",
                            off,
                            len,
                            bytes_data.len(),
                            i
                        ),
                    ));
                }
            }
        }
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown column type tag {}", col_type_tag),
            ));
        }
    }
    if pos != data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing bytes in column block",
        ));
    }
    Ok(())
}

/// Deserialize a single column block from raw bytes.
/// For Dictionary columns, pass the dictionary; for Bytes columns, pass ext_type.
/// An owned block whose bytes are mostly its blob becomes the column's bytes.
pub(crate) fn deserialize_column_block(
    block: Cow<'_, [u8]>,
    col_type_tag: u8,
    row_count: usize,
    dictionary: Option<Arc<[SmartString]>>,
    ext_type: DataType,
) -> io::Result<ColumnData> {
    let data = &*block;
    let block_len = data.len();
    let mut pos = 0;
    let column = match col_type_tag {
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
            let dict = dictionary.unwrap_or_else(|| Arc::from(Vec::<SmartString>::new()));
            // Validate dictionary IDs to prevent panics on corrupted volumes
            for (i, &id) in ids.iter().enumerate() {
                if !nulls[i] && (id as usize) >= dict.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "dictionary ID {} at row {} exceeds dictionary length {}",
                            id,
                            i,
                            dict.len()
                        ),
                    ));
                }
            }
            Ok(ColumnData::Dictionary {
                ids,
                dictionary: dict,
                nulls,
            })
        }
        COL_BYTES => {
            let nulls = read_nulls(data, &mut pos, row_count)?;
            let offset_count = read_u64(data, &mut pos)?;
            if offset_count != row_count as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "bytes offset count does not match row count",
                ));
            }
            if row_count > data.len().saturating_sub(pos) / 16 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated column block: bytes offsets",
                ));
            }
            let mut offsets = Vec::with_capacity(row_count);
            for _ in 0..row_count {
                let off = read_u64(data, &mut pos)?;
                let len = read_u64(data, &mut pos)?;
                offsets.push((off, len));
            }
            let data_len = read_u64(data, &mut pos)?;
            if data_len > data.len().saturating_sub(pos) as u64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "truncated column block: bytes data",
                ));
            }
            let end_pos = pos + data_len as usize;
            // Validate offsets to prevent panics on corrupted volumes
            for (i, &(off, len)) in offsets.iter().enumerate() {
                if !nulls[i] {
                    let end = off.checked_add(len).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("bytes offset overflow at row {}", i),
                        )
                    })?;
                    if end > data_len {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "bytes offset {}+{} exceeds data length {} at row {}",
                                off, len, data_len, i
                            ),
                        ));
                    }
                }
            }
            let blob = match block {
                // Its head of nulls and offsets stays, the offsets moved past it
                Cow::Owned(raw) if pos <= (end_pos - pos) / 8 => {
                    for (i, (off, _)) in offsets.iter_mut().enumerate() {
                        if !nulls[i] {
                            *off += pos as u64;
                        }
                    }
                    raw
                }
                block => block[pos..end_pos].to_vec(),
            };
            pos = end_pos;
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
    }?;
    if pos != block_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing bytes in column block",
        ));
    }
    Ok(column)
}

/// Metadata parsed from a V4 volume file (everything except column data).
pub(crate) struct VolumeMetadata {
    pub row_count: usize,
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
    let (kinds, dictionaries) = column_kinds(vol)?;
    serialize_volume_metadata_parts(&vol.meta, &kinds, &dictionaries)
}

/// The V5 metadata sections of a volume with decoded columns
pub(crate) fn v5_volume_sections(vol: &FrozenVolume) -> io::Result<Vec<V5Section>> {
    let (kinds, dictionaries) = column_kinds(vol)?;
    v5_metadata_sections(&vol.meta, &kinds, &dictionaries)
}

/// Every column's (storage type tag, extension type tag), and the text
/// columns' dictionaries in column order
#[allow(clippy::type_complexity)]
fn column_kinds(vol: &FrozenVolume) -> io::Result<(Vec<(u8, u8)>, Vec<&[SmartString]>)> {
    let col_count = vol.columns.len();
    let mut kinds = Vec::with_capacity(col_count);
    let mut dictionaries: Vec<&[SmartString]> = Vec::new();
    for i in 0..col_count {
        let col = vol.columns.get(i)?;
        kinds.push(match col {
            ColumnData::Int64 { .. } => (COL_INT64, 0),
            ColumnData::Float64 { .. } => (COL_FLOAT64, 0),
            ColumnData::TimestampNanos { .. } => (COL_TIMESTAMP, 0),
            ColumnData::Boolean { .. } => (COL_BOOLEAN, 0),
            ColumnData::Dictionary { .. } => (COL_DICTIONARY, 0),
            ColumnData::Bytes { ext_type, .. } => (COL_BYTES, *ext_type as u8),
        });
        if let ColumnData::Dictionary { dictionary, .. } = col {
            dictionaries.push(dictionary);
        }
    }
    Ok((kinds, dictionaries))
}

/// The metadata section from its parts, for a producer that has no
/// decoded columns: `kinds` is every column's (storage type tag, extension
/// type tag), `dictionaries` the text columns' dictionaries in column order
/// Serializes the metadata into a buffer of exactly its size: one pass
/// counts the bytes, the second writes them, so the buffer never grows
/// past what the file gets (the bloom filters alone outweigh the row ids).
pub(crate) fn serialize_volume_metadata_parts(
    meta: &super::writer::VolumeMeta,
    kinds: &[(u8, u8)],
    dictionaries: &[&[SmartString]],
) -> io::Result<Vec<u8>> {
    let mut counter = Counting(0);
    write_volume_metadata_parts(&mut counter, meta, kinds, dictionaries)?;
    let mut buf = Vec::with_capacity(counter.0);
    write_volume_metadata_parts(&mut buf, meta, kinds, dictionaries)?;
    debug_assert_eq!(buf.len(), counter.0);
    Ok(buf)
}

/// A writer that only counts
struct Counting(usize);

impl Write for Counting {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 += bytes.len();
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn write_volume_metadata_parts<W: Write>(
    buf: &mut W,
    meta: &super::writer::VolumeMeta,
    kinds: &[(u8, u8)],
    dictionaries: &[&[SmartString]],
) -> io::Result<()> {
    let col_count = kinds.len();
    let row_ids = &meta.row_ids;

    // Row count + col count
    buf.write_all(&(meta.row_count as u64).to_le_bytes())?;
    buf.write_all(&(col_count as u32).to_le_bytes())?;

    // Column directory: type(1) + flags(1) + extra(4) per column
    let mut dict_col_idx = 0usize;
    for (i, &(type_tag, ext_tag)) in kinds.iter().enumerate() {
        let sorted_flag = if meta.sorted_columns[i] {
            FLAG_SORTED
        } else {
            0
        };
        buf.write_all(&[type_tag])?;
        buf.write_all(&[sorted_flag])?;
        if type_tag == COL_DICTIONARY {
            let dictionary = dictionaries.get(dict_col_idx).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "text column without a dictionary",
                )
            })?;
            buf.write_all(&(dictionary.len() as u32).to_le_bytes())?;
            dict_col_idx += 1;
        } else if type_tag == COL_BYTES {
            buf.write_all(&(ext_tag as u32).to_le_bytes())?;
        } else {
            buf.write_all(&[0u8; 4])?;
        }
    }

    // Shared dictionary
    let shared_len: usize = dictionaries.iter().map(|d| d.len()).sum();
    buf.write_all(&(shared_len as u32).to_le_bytes())?;
    for s in dictionaries.iter().flat_map(|d| d.iter()) {
        let bytes = s.as_bytes();
        buf.write_all(&(bytes.len() as u32).to_le_bytes())?;
        buf.write_all(bytes)?;
    }

    // Row IDs (bulk — single memcpy on LE)
    write_i64_le(buf, row_ids)?;

    // Zone maps
    for zm in &meta.zone_maps {
        write_value(buf, &zm.min)?;
        write_value(buf, &zm.max)?;
        buf.write_all(&zm.null_count.to_le_bytes())?;
        buf.write_all(&zm.row_count.to_le_bytes())?;
    }

    // Bloom filters
    buf.write_all(&(meta.bloom_filters.len() as u32).to_le_bytes())?;
    for bf in &meta.bloom_filters {
        buf.write_all(&(bf.num_bits() as u64).to_le_bytes())?;
        buf.write_all(&((bf.bits().len() * 8) as u32).to_le_bytes())?;
        write_u64_le(buf, bf.bits())?;
    }

    // Stats
    buf.write_all(&meta.stats.total_rows.to_le_bytes())?;
    buf.write_all(&meta.stats.live_rows.to_le_bytes())?;
    buf.write_all(&(meta.stats.columns.len() as u32).to_le_bytes())?;
    for cs in &meta.stats.columns {
        buf.write_all(&cs.sum_int.to_le_bytes())?;
        buf.write_all(&cs.sum_float.to_le_bytes())?;
        buf.write_all(&cs.numeric_count.to_le_bytes())?;
        buf.write_all(&cs.non_null_count.to_le_bytes())?;
        write_value(buf, &cs.min)?;
        write_value(buf, &cs.max)?;
    }

    // Column names
    for name in &meta.column_names {
        let bytes = name.as_bytes();
        buf.write_all(&(bytes.len() as u32).to_le_bytes())?;
        buf.write_all(bytes)?;
    }

    // Column types
    for dt in &meta.column_types {
        buf.write_all(&[*dt as u8])?;
    }

    // Row groups
    buf.write_all(&(meta.row_groups.len() as u32).to_le_bytes())?;
    for rg in &meta.row_groups {
        buf.write_all(&rg.start_idx.to_le_bytes())?;
        buf.write_all(&rg.end_idx.to_le_bytes())?;
        for zm in &rg.zone_maps {
            write_value(buf, &zm.min)?;
            write_value(buf, &zm.max)?;
            buf.write_all(&zm.null_count.to_le_bytes())?;
            buf.write_all(&zm.row_count.to_le_bytes())?;
        }
    }

    Ok(())
}

fn check_metadata_count(data: &[u8], pos: usize, count: usize, width: usize) -> io::Result<()> {
    if count > (data.len() - pos) / width {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "V4 metadata count exceeds remaining bytes",
        ));
    }
    Ok(())
}

/// Deserialize volume metadata from V4 format bytes.
pub(crate) fn deserialize_volume_metadata(data: &[u8]) -> io::Result<VolumeMetadata> {
    let mut pos = 0;

    let row_count = usize::try_from(read_u64(data, &mut pos)?).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "V4 row count exceeds address space",
        )
    })?;
    let col_count = read_u32(data, &mut pos)? as usize;

    // Column directory
    check_metadata_count(data, pos, col_count, 6)?;
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
    let declared_dict_len = col_dict_counts.iter().try_fold(0usize, |sum, &count| {
        sum.checked_add(count as usize).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "V4 dictionary count overflow")
        })
    })?;
    if declared_dict_len != dict_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "V4 dictionary counts do not match shared dictionary length",
        ));
    }
    check_metadata_count(data, pos, dict_len, 4)?;
    let mut shared_dict = Vec::with_capacity(dict_len);
    for _ in 0..dict_len {
        let slen = read_u32(data, &mut pos)? as usize;
        if slen > data.len() - pos {
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
    check_metadata_count(data, pos, row_count, 8)?;
    let row_ids = read_i64_bulk(data, &mut pos, row_count)?;

    // Zone maps
    check_metadata_count(data, pos, col_count, 12)?;
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
    check_metadata_count(data, pos, num_blooms, 20)?;
    let mut bloom_filters = Vec::with_capacity(num_blooms);
    for _ in 0..num_blooms {
        let num_bits = usize::try_from(read_u64(data, &mut pos)?).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "V4 bloom size exceeds address space",
            )
        })?;
        let data_len = read_u32(data, &mut pos)? as usize;
        if data_len > data.len() - pos {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated V4 metadata: bloom filter",
            ));
        }
        let bits_bytes = &data[pos..pos + data_len];
        pos += data_len;
        if num_bits == 0 || !data_len.is_multiple_of(8) || num_bits.div_ceil(64) != data_len / 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid V4 bloom backing geometry",
            ));
        }
        bloom_filters.push(super::column::ColumnBloomFilter::from_parts(
            num_bits, bits_bytes,
        ));
    }

    // Stats
    let total_rows = read_u64(data, &mut pos)?;
    let live_rows = read_u64(data, &mut pos)?;
    let stats_col_count = read_u32(data, &mut pos)? as usize;
    if stats_col_count != col_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "V4 statistics column count mismatch",
        ));
    }
    check_metadata_count(data, pos, stats_col_count, 44)?;
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
    check_metadata_count(data, pos, col_count, 4)?;
    let mut column_names = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        let slen = read_u32(data, &mut pos)? as usize;
        if slen > data.len() - pos {
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
    let group_width = col_count
        .checked_mul(12)
        .and_then(|width| width.checked_add(8))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "V4 group width overflow"))?;
    check_metadata_count(data, pos, num_groups, group_width)?;
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

    if pos != data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "trailing bytes in V4 metadata",
        ));
    }

    let column_name_map = name_map(&column_names);

    Ok(VolumeMetadata {
        row_count,
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

/// Original and lowercase column names to positions
fn name_map(column_names: &[String]) -> ahash::AHashMap<SmartString, usize> {
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

// =============================================================================
// V5 metadata sections
// =============================================================================

pub(crate) const V5_COLUMNS: u32 = 4;
pub(crate) const V5_GROUPS: u32 = 5;
pub(crate) const V5_DICT: u32 = 6;
pub(crate) const V5_ROW_IDS: u32 = 7;
pub(crate) const V5_ZONES: u32 = 8;
pub(crate) const V5_BLOOMS: u32 = 9;
pub(crate) const V5_STATS: u32 = 10;

/// Column entries (type, flags, extension, encoding, kind, data type,
/// dictionary size, name length), then the names
const V5_COLUMN_ENTRY: usize = 16;
/// Group entries: first row u64, rows u32, reserved u32
const V5_GROUP_ENTRY: usize = 16;
/// Every row id as i64
const ROW_IDS_RAW: u8 = 1;
/// Ascending ids with no gap: the first and the count
const ROW_IDS_RANGE: u8 = 2;
/// Bloom hash: FNV-1a 64 over a type tag and the value's LE bytes, three
/// probes, bits as LE u64 words
const BLOOM_FNV1A64_K3: u8 = 1;
/// Bloom section header: hash id, probes, reserved, column count, CRC of
/// the entries, entry size; one entry per column (bits, words, CRC of
/// the words, offset from the section's start), then the words
const V5_BLOOM_HEAD: usize = 16;
const V5_BLOOM_ENTRY: usize = 24;

/// One metadata section as written: its kind, whether a reader must
/// understand it, its encoding, whether to LZ4 it, and its bytes
pub(crate) struct V5Section {
    pub kind: u32,
    pub required: bool,
    pub encoding: u8,
    pub compress: bool,
    pub bytes: Vec<u8>,
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("V5 metadata: {msg}"))
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The metadata of a volume as V5 sections, from the same parts the V4
/// metadata is written from
pub(crate) fn v5_metadata_sections(
    meta: &super::writer::VolumeMeta,
    kinds: &[(u8, u8)],
    dictionaries: &[&[SmartString]],
) -> io::Result<Vec<V5Section>> {
    let col_count = kinds.len();
    let section = |kind, required, encoding, compress, bytes| V5Section {
        kind,
        required,
        encoding,
        compress,
        bytes,
    };
    let mut sections = Vec::with_capacity(7);

    let mut columns = Vec::with_capacity(16 + col_count * (V5_COLUMN_ENTRY + 16));
    columns.extend_from_slice(&(meta.row_count as u64).to_le_bytes());
    columns.extend_from_slice(&(col_count as u32).to_le_bytes());
    columns.extend_from_slice(&(V5_COLUMN_ENTRY as u32).to_le_bytes());
    let mut dict_col = 0usize;
    for (i, &(type_tag, ext_tag)) in kinds.iter().enumerate() {
        let flags = if meta.sorted_columns[i] {
            FLAG_SORTED
        } else {
            0
        };
        let dict_count = if type_tag == COL_DICTIONARY {
            let d = dictionaries.get(dict_col).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "text column without a dictionary",
                )
            })?;
            dict_col += 1;
            d.len() as u32
        } else {
            0
        };
        let dtype = meta.column_types.get(i).copied().unwrap_or(DataType::Null) as u8;
        // Encoding 1 is V4's cells; kind 0 a data column
        columns.extend_from_slice(&[type_tag, flags, ext_tag, 1, 0, dtype, 0, 0]);
        columns.extend_from_slice(&dict_count.to_le_bytes());
        columns.extend_from_slice(&(meta.column_names[i].len() as u32).to_le_bytes());
    }
    for name in &meta.column_names {
        columns.extend_from_slice(name.as_bytes());
    }
    sections.push(section(V5_COLUMNS, true, 1, false, columns));

    let group_count = meta.row_count.div_ceil(meta.group_rows);
    let mut groups = Vec::with_capacity(8 + group_count * V5_GROUP_ENTRY);
    groups.extend_from_slice(&(group_count as u32).to_le_bytes());
    groups.extend_from_slice(&(V5_GROUP_ENTRY as u32).to_le_bytes());
    for g in 0..group_count {
        let start = g * meta.group_rows;
        let rows = meta.group_rows.min(meta.row_count - start);
        groups.extend_from_slice(&(start as u64).to_le_bytes());
        groups.extend_from_slice(&(rows as u32).to_le_bytes());
        groups.extend_from_slice(&[0u8; 4]);
    }
    sections.push(section(V5_GROUPS, true, 1, false, groups));

    let strings: usize = dictionaries.iter().map(|d| d.len()).sum();
    if strings > 0 {
        let text: usize = dictionaries
            .iter()
            .flat_map(|d| d.iter())
            .map(|s| s.len())
            .sum();
        let mut dict = Vec::with_capacity(4 + strings * 4 + text);
        dict.extend_from_slice(&(strings as u32).to_le_bytes());
        let mut end = 0u64;
        for s in dictionaries.iter().flat_map(|d| d.iter()) {
            end += s.len() as u64;
            let end = u32::try_from(end).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "V5 dictionary over 4 GiB")
            })?;
            dict.extend_from_slice(&end.to_le_bytes());
        }
        for s in dictionaries.iter().flat_map(|d| d.iter()) {
            dict.extend_from_slice(s.as_bytes());
        }
        sections.push(section(V5_DICT, true, 1, true, dict));
    }

    let ids = &meta.row_ids;
    let contiguous = !ids.is_empty() && ids.windows(2).all(|w| w[0].checked_add(1) == Some(w[1]));
    if contiguous {
        let mut b = Vec::with_capacity(16);
        b.extend_from_slice(&ids[0].to_le_bytes());
        b.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        sections.push(section(V5_ROW_IDS, true, ROW_IDS_RANGE, false, b));
    } else {
        let mut b = Vec::with_capacity(8 + ids.len() * 8);
        b.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        write_i64_le(&mut b, ids)?;
        sections.push(section(V5_ROW_IDS, true, ROW_IDS_RAW, true, b));
    }

    let mut zones = Vec::new();
    let put_zone = |b: &mut Vec<u8>, zm: &ZoneMap| -> io::Result<()> {
        write_value(b, &zm.min)?;
        write_value(b, &zm.max)?;
        b.extend_from_slice(&zm.null_count.to_le_bytes());
        b.extend_from_slice(&zm.row_count.to_le_bytes());
        Ok(())
    };
    for zm in &meta.zone_maps {
        put_zone(&mut zones, zm)?;
    }
    zones.extend_from_slice(&(meta.row_groups.len() as u32).to_le_bytes());
    for rg in &meta.row_groups {
        for zm in &rg.zone_maps {
            put_zone(&mut zones, zm)?;
        }
    }
    sections.push(section(V5_ZONES, true, 1, true, zones));

    // A bytes column's values all hash alike, so its filter rejects
    // nothing and is not written
    let filters: Vec<Option<&super::column::ColumnBloomFilter>> = (0..col_count)
        .map(|i| {
            (kinds[i].0 != COL_BYTES)
                .then(|| meta.bloom_filters.get(i))
                .flatten()
        })
        .collect();
    let head = V5_BLOOM_HEAD + col_count * V5_BLOOM_ENTRY;
    let words: usize = filters.iter().flatten().map(|bf| bf.bits().len()).sum();
    let mut blooms = Vec::with_capacity(head + words * 8);
    blooms.extend_from_slice(&[BLOOM_FNV1A64_K3, 3, 0, 0]);
    blooms.extend_from_slice(&(col_count as u32).to_le_bytes());
    blooms.extend_from_slice(&[0u8; 4]);
    blooms.extend_from_slice(&(V5_BLOOM_ENTRY as u32).to_le_bytes());
    let mut offset = head as u64;
    let mut words_bytes = Vec::new();
    for bf in &filters {
        let (bits, words, crc) = match bf {
            Some(bf) => {
                words_bytes.clear();
                write_u64_le(&mut words_bytes, bf.bits())?;
                let crc = super::v5::checksum(&words_bytes);
                (bf.num_bits() as u64, bf.bits().len() as u32, crc)
            }
            None => (0, 0, 0),
        };
        blooms.extend_from_slice(&bits.to_le_bytes());
        blooms.extend_from_slice(&words.to_le_bytes());
        blooms.extend_from_slice(&crc.to_le_bytes());
        blooms.extend_from_slice(&offset.to_le_bytes());
        offset += u64::from(words) * 8;
    }
    let entries_crc = super::v5::checksum(&blooms[V5_BLOOM_HEAD..head]);
    blooms[8..12].copy_from_slice(&entries_crc.to_le_bytes());
    for bf in filters.iter().flatten() {
        write_u64_le(&mut blooms, bf.bits())?;
    }
    sections.push(section(V5_BLOOMS, false, 1, false, blooms));

    let mut stats = Vec::with_capacity(20 + col_count * 64);
    stats.extend_from_slice(&meta.stats.total_rows.to_le_bytes());
    stats.extend_from_slice(&meta.stats.live_rows.to_le_bytes());
    stats.extend_from_slice(&(meta.stats.columns.len() as u32).to_le_bytes());
    for cs in &meta.stats.columns {
        stats.extend_from_slice(&cs.sum_int.to_le_bytes());
        stats.extend_from_slice(&cs.sum_float.to_le_bytes());
        stats.extend_from_slice(&cs.numeric_count.to_le_bytes());
        stats.extend_from_slice(&cs.non_null_count.to_le_bytes());
        write_value(&mut stats, &cs.min)?;
        write_value(&mut stats, &cs.max)?;
    }
    sections.push(section(V5_STATS, true, 1, false, stats));
    Ok(sections)
}

/// A V5 volume's metadata and its group directory's (first, end) rows
pub(crate) struct V5Metadata {
    pub meta: VolumeMetadata,
    pub bounds: Vec<(u32, u32)>,
}

/// The volume metadata from its V5 sections, each (kind, encoding, bytes)
pub(crate) fn v5_metadata_from_sections(sections: &[(u32, u8, Vec<u8>)]) -> io::Result<V5Metadata> {
    let get = |kind: u32| {
        sections
            .iter()
            .find(|s| s.0 == kind)
            .map(|s| (s.1, s.2.as_slice()))
    };

    let (enc, data) = get(V5_COLUMNS).ok_or_else(|| invalid("no column directory"))?;
    if enc != 1 || data.len() < 16 {
        return Err(invalid("column directory"));
    }
    let mut pos = 0;
    let row_count = usize::try_from(read_u64(data, &mut pos)?).map_err(|_| invalid("row count"))?;
    let col_count = read_u32(data, &mut pos)? as usize;
    let entry = read_u32(data, &mut pos)? as usize;
    if entry < V5_COLUMN_ENTRY {
        return Err(invalid("column entry size"));
    }
    check_metadata_count(data, pos, col_count, entry)?;
    let mut col_type_tags = Vec::with_capacity(col_count);
    let mut col_ext_types = Vec::with_capacity(col_count);
    let mut col_sorted = Vec::with_capacity(col_count);
    let mut col_dict_counts = Vec::with_capacity(col_count);
    let mut column_types = Vec::with_capacity(col_count);
    let mut name_lens = Vec::with_capacity(col_count);
    for i in 0..col_count {
        let e = &data[pos + i * entry..pos + (i + 1) * entry];
        // A cell encoding, column kind or data type this reader lacks
        // would be read as something else
        let dtype = DataType::from_u8(e[5]).ok_or_else(|| invalid("unknown data type"))?;
        if e[3] != 1 || e[4] != 0 || !(COL_INT64..=COL_BYTES).contains(&e[0]) {
            return Err(invalid("unknown column type, encoding or kind"));
        }
        col_type_tags.push(e[0]);
        col_sorted.push(e[1] & FLAG_SORTED != 0);
        col_ext_types.push(if e[0] == COL_BYTES { e[2] } else { 0 });
        column_types.push(dtype);
        col_dict_counts.push(u32_at(e, 8));
        name_lens.push(u32_at(e, 12) as usize);
    }
    pos += col_count * entry;
    let mut column_names = Vec::with_capacity(col_count);
    for len in name_lens {
        if len > data.len() - pos {
            return Err(invalid("column name"));
        }
        let s = std::str::from_utf8(&data[pos..pos + len]).map_err(|_| invalid("column name"))?;
        column_names.push(s.to_string());
        pos += len;
    }

    let (enc, data) = get(V5_GROUPS).ok_or_else(|| invalid("no group directory"))?;
    if enc != 1 || data.len() < 8 {
        return Err(invalid("group directory"));
    }
    let (group_count, entry) = (u32_at(data, 0) as usize, u32_at(data, 4) as usize);
    if entry < V5_GROUP_ENTRY {
        return Err(invalid("group entry size"));
    }
    check_metadata_count(data, 8, group_count, entry)?;
    let mut bounds = Vec::with_capacity(group_count);
    for g in 0..group_count {
        let mut p = 8 + g * entry;
        let start = read_u64(data, &mut p)?;
        let rows = u64::from(read_u32(data, &mut p)?);
        let end = start
            .checked_add(rows)
            .ok_or_else(|| invalid("group bounds"))?;
        let narrow = |v: u64| u32::try_from(v).map_err(|_| invalid("group bounds"));
        bounds.push((narrow(start)?, narrow(end)?));
    }

    let strings = col_dict_counts
        .iter()
        .try_fold(0usize, |sum, &c| sum.checked_add(c as usize))
        .ok_or_else(|| invalid("dictionary count"))?;
    let mut shared_dict = Vec::with_capacity(strings);
    if strings > 0 {
        let (enc, data) = get(V5_DICT).ok_or_else(|| invalid("no dictionary"))?;
        if enc != 1 || data.len() < 4 || u32_at(data, 0) as usize != strings {
            return Err(invalid("dictionary"));
        }
        check_metadata_count(data, 4, strings, 4)?;
        let text = std::str::from_utf8(&data[4 + strings * 4..])
            .map_err(|_| invalid("dictionary text"))?;
        let mut start = 0usize;
        for i in 0..strings {
            let end = u32_at(data, 4 + i * 4) as usize;
            if end < start || end > text.len() || !text.is_char_boundary(end) {
                return Err(invalid("dictionary bounds"));
            }
            shared_dict.push(SmartString::from(&text[start..end]));
            start = end;
        }
        if start != text.len() {
            return Err(invalid("dictionary trailing text"));
        }
    }

    let (enc, data) = get(V5_ROW_IDS).ok_or_else(|| invalid("no row ids"))?;
    let mut pos = 0;
    let row_ids = match enc {
        ROW_IDS_RANGE => {
            let first = read_u64(data, &mut pos)? as i64;
            let count = read_u64(data, &mut pos)?;
            if count != row_count as u64
                || first.checked_add(count as i64).is_none()
                || pos != data.len()
            {
                return Err(invalid("row id range"));
            }
            (0..count as i64).map(|i| first + i).collect()
        }
        ROW_IDS_RAW => {
            let count = read_u64(data, &mut pos)?;
            if count != row_count as u64 {
                return Err(invalid("row id count"));
            }
            check_metadata_count(data, pos, row_count, 8)?;
            let ids = read_i64_bulk(data, &mut pos, row_count)?;
            if pos != data.len() {
                return Err(invalid("row id trailing bytes"));
            }
            ids
        }
        _ => return Err(invalid("row id encoding")),
    };

    let (enc, data) = get(V5_ZONES).ok_or_else(|| invalid("no zone maps"))?;
    if enc != 1 {
        return Err(invalid("zone map encoding"));
    }
    let mut pos = 0;
    let read_zone = |pos: &mut usize| -> io::Result<ZoneMap> {
        Ok(ZoneMap {
            min: read_value(data, pos)?,
            max: read_value(data, pos)?,
            null_count: read_u32(data, pos)?,
            row_count: read_u32(data, pos)?,
        })
    };
    check_metadata_count(data, pos, col_count, 12)?;
    let mut zone_maps = Vec::with_capacity(col_count);
    for _ in 0..col_count {
        zone_maps.push(read_zone(&mut pos)?);
    }
    let zoned = read_u32(data, &mut pos)? as usize;
    // Groups have their zone maps exactly when there is more than one
    if zoned != if group_count > 1 { group_count } else { 0 } {
        return Err(invalid("group zone map count"));
    }
    check_metadata_count(data, pos, zoned, col_count * 12)?;
    let mut row_groups = Vec::with_capacity(zoned);
    for &(start_idx, end_idx) in bounds.iter().take(zoned) {
        let mut zms = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            zms.push(read_zone(&mut pos)?);
        }
        row_groups.push(super::column::RowGroupMeta {
            start_idx,
            end_idx,
            zone_maps: zms,
        });
    }
    if pos != data.len() {
        return Err(invalid("zone map trailing bytes"));
    }

    let bloom_filters = match get(V5_BLOOMS) {
        Some((enc, data)) => v5_blooms_from_section(enc, data, col_count)?,
        // A reader without the section prunes nothing
        None => (0..col_count).map(|_| accept_all_filter()).collect(),
    };

    let (enc, data) = get(V5_STATS).ok_or_else(|| invalid("no statistics"))?;
    if enc != 1 {
        return Err(invalid("statistics encoding"));
    }
    let mut pos = 0;
    let total_rows = read_u64(data, &mut pos)?;
    let live_rows = read_u64(data, &mut pos)?;
    let stats_cols = read_u32(data, &mut pos)? as usize;
    if stats_cols != col_count {
        return Err(invalid("statistics column count"));
    }
    check_metadata_count(data, pos, stats_cols, 44)?;
    let mut stat_columns = Vec::with_capacity(stats_cols);
    for _ in 0..stats_cols {
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
    if pos != data.len() {
        return Err(invalid("statistics trailing bytes"));
    }

    let column_name_map = name_map(&column_names);
    Ok(V5Metadata {
        meta: VolumeMetadata {
            row_count,
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
        },
        bounds,
    })
}

/// A filter that rejects nothing: every bit set
fn accept_all_filter() -> super::column::ColumnBloomFilter {
    super::column::ColumnBloomFilter::from_parts(64, &[0xFF; 8])
}

/// Every column's filter of a V5 bloom section; a column the section has
/// no filter for gets one that rejects nothing
fn v5_blooms_from_section(
    enc: u8,
    data: &[u8],
    col_count: usize,
) -> io::Result<Vec<super::column::ColumnBloomFilter>> {
    if enc != 1 || data.len() < V5_BLOOM_HEAD || data[0] != BLOOM_FNV1A64_K3 || data[1] != 3 {
        return Err(invalid("bloom hash or probes"));
    }
    let (count, entry) = (u32_at(data, 4) as usize, u32_at(data, 12) as usize);
    if count != col_count || entry < V5_BLOOM_ENTRY {
        return Err(invalid("bloom header"));
    }
    check_metadata_count(data, V5_BLOOM_HEAD, count, entry)?;
    let head = V5_BLOOM_HEAD + count * entry;
    if super::v5::checksum(&data[V5_BLOOM_HEAD..head]) != u32_at(data, 8) {
        return Err(invalid("bloom header CRC mismatch"));
    }
    (0..count)
        .map(|i| {
            let mut p = V5_BLOOM_HEAD + i * entry;
            let num_bits = read_u64(data, &mut p)?;
            let words = read_u32(data, &mut p)? as usize;
            let crc = read_u32(data, &mut p)?;
            let offset =
                usize::try_from(read_u64(data, &mut p)?).map_err(|_| invalid("bloom offset"))?;
            if words == 0 {
                return if num_bits == 0 {
                    Ok(accept_all_filter())
                } else {
                    Err(invalid("bloom geometry"))
                };
            }
            let end = words
                .checked_mul(8)
                .and_then(|n| n.checked_add(offset))
                .filter(|&end| offset >= head && end <= data.len())
                .ok_or_else(|| invalid("bloom bits outside the section"))?;
            if num_bits.div_ceil(64) != words as u64 {
                return Err(invalid("bloom geometry"));
            }
            let bits = &data[offset..end];
            if super::v5::checksum(bits) != crc {
                return Err(invalid("bloom CRC mismatch"));
            }
            let num_bits = usize::try_from(num_bits).map_err(|_| invalid("bloom size"))?;
            Ok(super::column::ColumnBloomFilter::from_parts(num_bits, bits))
        })
        .collect()
}
