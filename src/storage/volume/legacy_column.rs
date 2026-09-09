// Copyright 2026 Stoolap Contributors
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

//! Inactive, allocation-free adapter for a decoded V4 column block.
//!
//! The caller owns the seekable spool and all buffers. Before construction it
//! must finish decoding and outer checksum validation, and keep this region
//! immutable (including through aliases to a file). An exclusive Rust borrow
//! prevents reuse through this handle, not mutation through other file handles.
//! This adapter validates the column layout; it does not establish authenticity,
//! manage files, enforce an engine memory budget, or replace active readers.
//!
//! V4 Bytes offsets need not be ordered, disjoint, or packed. NULL dictionary
//! IDs may be out of range. Both are intentional compatibility rules. Extension
//! bytes are opaque, just as in the existing column decoder. Dictionary cells
//! expose volume-local IDs: paged string resolution and group-local re-encoding
//! belong to the future metadata/dictionary adapter.
//!
//! Range planning caps payload bytes and row count separately. Callers reading
//! several columns choose a common physical range from their plans, then gather
//! that exact range into reusable buffers. Buffer-too-small preflight never
//! changes row/payload output; it may seek/read the spool and change scratch.
//! Any I/O/format failure or panic after I/O starts permanently aborts the handle.
//! After a failed gather, output is provisional and no view is returned.

use std::fmt;
use std::io::{self, Read, Seek, SeekFrom};
use std::num::NonZeroUsize;

use crate::core::DataType;

use super::format::{
    COL_BOOLEAN, COL_BYTES, COL_DICTIONARY, COL_FLOAT64, COL_INT64, COL_TIMESTAMP,
};

pub const MAX_ROWS: u32 = 65_536;
pub const MIN_SCRATCH_BYTES: usize = 17;
pub const MAX_SCRATCH_BYTES: usize = 65_536;

type Result<T> = std::result::Result<T, ColumnError>;

#[derive(Debug)]
pub enum ColumnError {
    InvalidScratch,
    InvalidEncoding,
    InvalidRowCount,
    InvalidRange,
    LengthOverflow,
    LengthMismatch,
    TruncatedInput,
    InvalidFlag,
    InvalidOffset,
    InvalidDictionaryId,
    InvalidReadCount,
    InvalidSeekPosition,
    BufferTooSmall { rows: usize, payload_bytes: u64 },
    Aborted,
    Io(io::Error),
}

impl fmt::Display for ColumnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScratch => f.write_str("column scratch must contain 17..=65536 bytes"),
            Self::InvalidEncoding => f.write_str("invalid V4 column encoding"),
            Self::InvalidRowCount => f.write_str("V4 physical group exceeds 65536 rows"),
            Self::InvalidRange => f.write_str("invalid V4 column row range"),
            Self::LengthOverflow => f.write_str("V4 column length overflow"),
            Self::LengthMismatch => f.write_str("V4 column length/count mismatch"),
            Self::TruncatedInput => f.write_str("truncated V4 column spool"),
            Self::InvalidFlag => f.write_str("invalid V4 null/boolean flag"),
            Self::InvalidOffset => f.write_str("V4 Bytes offset exceeds blob length"),
            Self::InvalidDictionaryId => f.write_str("V4 dictionary ID exceeds dictionary length"),
            Self::InvalidReadCount => f.write_str("spool reader exceeded requested byte count"),
            Self::InvalidSeekPosition => f.write_str("spool seek returned an unexpected position"),
            Self::BufferTooSmall {
                rows,
                payload_bytes,
            } => {
                write!(
                    f,
                    "column range needs {rows} row slots and {payload_bytes} payload bytes"
                )
            }
            Self::Aborted => f.write_str("column spool handle is aborted"),
            Self::Io(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ColumnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for ColumnError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// Directory information already obtained from the volume metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    Int64,
    Float64,
    TimestampNanos,
    Boolean,
    Dictionary { entries: u32 },
    Bytes { data_type: DataType },
}

impl Encoding {
    pub fn from_directory(tag: u8, extra: u32) -> Result<Self> {
        Ok(match tag {
            COL_INT64 => Self::Int64,
            COL_FLOAT64 => Self::Float64,
            COL_TIMESTAMP => Self::TimestampNanos,
            COL_BOOLEAN => Self::Boolean,
            COL_DICTIONARY => Self::Dictionary { entries: extra },
            COL_BYTES => Self::Bytes {
                data_type: u8::try_from(extra)
                    .ok()
                    .and_then(DataType::from_u8)
                    .ok_or(ColumnError::InvalidEncoding)?,
            },
            _ => return Err(ColumnError::InvalidEncoding),
        })
    }

    fn width(self) -> usize {
        match self {
            Self::Int64 | Self::Float64 | Self::TimestampNanos => 8,
            Self::Boolean => 1,
            Self::Dictionary { .. } => 4,
            Self::Bytes { .. } => 16,
        }
    }

    fn data_type(self) -> DataType {
        match self {
            Self::Int64 => DataType::Integer,
            Self::Float64 => DataType::Float,
            Self::TimestampNanos => DataType::Timestamp,
            Self::Boolean => DataType::Boolean,
            Self::Dictionary { .. } => DataType::Text,
            Self::Bytes { data_type } => data_type,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ColumnSpec {
    pub encoding: Encoding,
    pub row_count: u32,
    pub offset: u64,
    pub decoded_len: u64,
}

/// All positions are physical row positions within this one V4 group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangePlan {
    pub start: u32,
    pub end: u32,
    pub payload_bytes: u64,
}

/// Caller-owned row metadata. Fields are private so a returned view always has
/// checked payload bounds. No references into the spool escape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RowSlot {
    // None is NULL; offset+1 also represents a non-NULL empty value at byte 0.
    offset: Option<NonZeroUsize>,
    len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CellRef<'a> {
    Null(DataType),
    Int64(i64),
    Float64(f64),
    TimestampNanos(i64),
    Boolean(bool),
    DictionaryId(u32),
    Bytes {
        data_type: DataType,
        bytes: &'a [u8],
    },
}

pub struct ColumnRange<'a> {
    encoding: Encoding,
    start: u32,
    rows: &'a [RowSlot],
    payload: &'a [u8],
}

impl<'a> ColumnRange<'a> {
    pub fn start(&self) -> u32 {
        self.start
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn payload_bytes(&self) -> usize {
        self.payload.len()
    }
    pub fn get(&self, row: usize) -> Option<CellRef<'a>> {
        let slot = self.rows.get(row)?;
        let Some(offset) = slot.offset else {
            return Some(CellRef::Null(self.encoding.data_type()));
        };
        let offset = offset.get() - 1;
        let bytes = &self.payload[offset..offset + slot.len];
        Some(match self.encoding {
            Encoding::Int64 => CellRef::Int64(i64::from_le_bytes(bytes.try_into().unwrap())),
            Encoding::Float64 => CellRef::Float64(f64::from_bits(u64::from_le_bytes(
                bytes.try_into().unwrap(),
            ))),
            Encoding::TimestampNanos => {
                CellRef::TimestampNanos(i64::from_le_bytes(bytes.try_into().unwrap()))
            }
            Encoding::Boolean => CellRef::Boolean(bytes[0] != 0),
            Encoding::Dictionary { .. } => {
                CellRef::DictionaryId(u32::from_le_bytes(bytes.try_into().unwrap()))
            }
            Encoding::Bytes { data_type } => CellRef::Bytes { data_type, bytes },
        })
    }
}

/// A fixed-size layout proof plus an exclusive borrow of caller backing.
pub struct ValidatedColumn<'a, R> {
    spool: &'a mut R,
    spec: ColumnSpec,
    lanes: u64,
    blob: u64,
    blob_len: u64,
    aborted: bool,
}

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).ok_or(ColumnError::LengthOverflow)
}

fn check_scratch(scratch: &[u8]) -> Result<()> {
    if !(MIN_SCRATCH_BYTES..=MAX_SCRATCH_BYTES).contains(&scratch.len()) {
        return Err(ColumnError::InvalidScratch);
    }
    Ok(())
}

fn seek<R: Seek>(spool: &mut R, position: SeekFrom) -> Result<u64> {
    loop {
        match spool.seek(position) {
            Ok(actual) => {
                if matches!(position, SeekFrom::Start(expected) if expected != actual) {
                    return Err(ColumnError::InvalidSeekPosition);
                }
                return Ok(actual);
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
}

fn read_at<R: Read + Seek>(spool: &mut R, offset: u64, mut out: &mut [u8]) -> Result<()> {
    if out.is_empty() {
        return Ok(());
    }
    seek(spool, SeekFrom::Start(offset))?;
    while !out.is_empty() {
        match spool.read(out) {
            Ok(0) => return Err(ColumnError::TruncatedInput),
            Ok(count) if count > out.len() => return Err(ColumnError::InvalidReadCount),
            Ok(count) => out = &mut out[count..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn u64_at(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().unwrap())
}

impl<'a, R: Read + Seek> ValidatedColumn<'a, R> {
    pub fn new(spool: &'a mut R, spec: ColumnSpec, scratch: &mut [u8]) -> Result<Self> {
        check_scratch(scratch)?;
        if spec.row_count > MAX_ROWS {
            return Err(ColumnError::InvalidRowCount);
        }
        let end = add(spec.offset, spec.decoded_len)?;
        let mut lanes = add(spec.offset, spec.row_count as u64)?;
        let is_bytes = matches!(spec.encoding, Encoding::Bytes { .. });
        let (blob, blob_len) = if is_bytes {
            // Prove header/table bounds before any spool read.
            let minimum = spec.row_count as u64 * 17 + 16;
            if spec.decoded_len < minimum {
                return Err(ColumnError::LengthMismatch);
            }
            read_at(spool, lanes, &mut scratch[..8])?;
            if u64_at(scratch) != spec.row_count as u64 {
                return Err(ColumnError::LengthMismatch);
            }
            lanes += 8;
            let length_position = add(lanes, spec.row_count as u64 * 16)?;
            read_at(spool, length_position, &mut scratch[..8])?;
            let blob_len = u64_at(scratch);
            let blob = add(length_position, 8)?;
            if add(blob, blob_len)? != end {
                return Err(ColumnError::LengthMismatch);
            }
            (blob, blob_len)
        } else {
            if spec.decoded_len != spec.row_count as u64 * (spec.encoding.width() as u64 + 1) {
                return Err(ColumnError::LengthMismatch);
            }
            (0, 0)
        };
        if seek(spool, SeekFrom::End(0))? < end {
            return Err(ColumnError::TruncatedInput);
        }
        let mut column = Self {
            spool,
            spec,
            lanes,
            blob,
            blob_len,
            aborted: true,
        };
        let mut row = 0;
        while row < spec.row_count {
            let count = column.read_records(row, spec.row_count, scratch)?;
            for index in 0..count {
                column.record(scratch, count, index)?;
            }
            row += count as u32;
        }
        column.aborted = false;
        Ok(column)
    }

    pub fn spec(&self) -> ColumnSpec {
        self.spec
    }
    pub fn is_aborted(&self) -> bool {
        self.aborted
    }

    fn check_range(&self, start: u32, end: u32, scratch: &[u8]) -> Result<()> {
        if self.aborted {
            return Err(ColumnError::Aborted);
        }
        check_scratch(scratch)?;
        if start > end || end > self.spec.row_count {
            return Err(ColumnError::InvalidRange);
        }
        Ok(())
    }

    // Scratch holds a batch of null flags followed by fixed-width wire records.
    fn read_records(&mut self, start: u32, end: u32, scratch: &mut [u8]) -> Result<usize> {
        let width = self.spec.encoding.width();
        let count = ((end - start) as usize).min(scratch.len() / (width + 1));
        read_at(
            self.spool,
            self.spec.offset + start as u64,
            &mut scratch[..count],
        )?;
        read_at(
            self.spool,
            self.lanes + start as u64 * width as u64,
            &mut scratch[count..count * (width + 1)],
        )?;
        Ok(count)
    }

    // Returns (NULL, source-relative offset, payload length). NULL Bytes values
    // retain offset validation but need no payload reads or output space.
    fn record(&self, scratch: &[u8], count: usize, index: usize) -> Result<(bool, u64, u64)> {
        let flag = scratch[index];
        if flag > 1 {
            return Err(ColumnError::InvalidFlag);
        }
        let width = self.spec.encoding.width();
        let lane = &scratch[count + index * width..count + (index + 1) * width];
        match self.spec.encoding {
            Encoding::Boolean if lane[0] > 1 => return Err(ColumnError::InvalidFlag),
            Encoding::Dictionary { entries }
                if flag == 0 && u32::from_le_bytes(lane.try_into().unwrap()) >= entries =>
            {
                return Err(ColumnError::InvalidDictionaryId);
            }
            Encoding::Bytes { .. } => {
                let offset = u64_at(lane);
                let len = u64_at(&lane[8..]);
                if offset
                    .checked_add(len)
                    .is_none_or(|end| end > self.blob_len)
                {
                    return Err(ColumnError::InvalidOffset);
                }
                return Ok((flag != 0, offset, if flag == 0 { len } else { 0 }));
            }
            _ => {}
        }
        Ok((flag != 0, 0, width as u64))
    }

    /// Largest prefix fitting a caller's payload cap. A first value that does
    /// not fit returns BufferTooSmall with its exact requirement, never an
    /// implicit allocation or a non-progressing empty range. An empty requested
    /// range is valid, as are arbitrarily many zero-byte/NULL Bytes cells.
    pub fn plan_range(
        &mut self,
        start: u32,
        end: u32,
        payload_capacity: usize,
        scratch: &mut [u8],
    ) -> Result<RangePlan> {
        self.check_range(start, end, scratch)?;
        self.aborted = true;
        let result = self.plan(start, end, payload_capacity as u64, scratch);
        if result.is_ok() || matches!(result, Err(ColumnError::BufferTooSmall { .. })) {
            self.aborted = false;
        }
        result
    }

    fn plan(&mut self, start: u32, end: u32, cap: u64, scratch: &mut [u8]) -> Result<RangePlan> {
        let mut row = start;
        let mut bytes = 0u64;
        while row < end {
            let count = self.read_records(row, end, scratch)?;
            for index in 0..count {
                let (_, _, len) = self.record(scratch, count, index)?;
                let next = add(bytes, len)?;
                if next > cap {
                    let matched_end = row + index as u32;
                    if matched_end == start {
                        return Err(ColumnError::BufferTooSmall {
                            rows: 1,
                            payload_bytes: len,
                        });
                    }
                    return Ok(RangePlan {
                        start,
                        end: matched_end,
                        payload_bytes: bytes,
                    });
                }
                bytes = next;
            }
            row += count as u32;
        }
        Ok(RangePlan {
            start,
            end,
            payload_bytes: bytes,
        })
    }

    /// Gather exactly the requested physical range. Fixed lanes are copied in
    /// batches; adjacent Bytes spans are coalesced within a metadata batch.
    /// The supplied output may be larger than needed; its unused tail is kept.
    pub fn read_range<'out>(
        &mut self,
        start: u32,
        end: u32,
        rows: &'out mut [RowSlot],
        payload: &'out mut [u8],
        scratch: &mut [u8],
    ) -> Result<ColumnRange<'out>> {
        self.check_range(start, end, scratch)?;
        let count = (end - start) as usize;
        if rows.len() < count {
            return Err(ColumnError::BufferTooSmall {
                rows: count,
                payload_bytes: 0,
            });
        }
        self.aborted = true;
        // Exact preflight before touching external output. Use u64's full cap
        // here to return the complete requirement rather than a partial plan.
        let plan = self.plan(start, end, u64::MAX, scratch)?;
        if plan.payload_bytes > payload.len() as u64 {
            self.aborted = false;
            return Err(ColumnError::BufferTooSmall {
                rows: count,
                payload_bytes: plan.payload_bytes,
            });
        }
        let payload_bytes = plan.payload_bytes as usize;
        self.gather(
            start,
            end,
            &mut rows[..count],
            &mut payload[..payload_bytes],
            scratch,
        )?;
        self.aborted = false;
        Ok(ColumnRange {
            encoding: self.spec.encoding,
            start,
            rows: &rows[..count],
            payload: &payload[..payload_bytes],
        })
    }

    fn gather(
        &mut self,
        start: u32,
        end: u32,
        rows: &mut [RowSlot],
        payload: &mut [u8],
        scratch: &mut [u8],
    ) -> Result<()> {
        let mut row = start;
        let mut used = 0usize;
        while row < end {
            let count = self.read_records(row, end, scratch)?;
            let width = self.spec.encoding.width();
            let mut run_source = 0;
            let mut run_start = used;
            let mut run_len = 0;
            for index in 0..count {
                let (is_null, offset, len) = self.record(scratch, count, index)?;
                let len = usize::try_from(len).map_err(|_| ColumnError::LengthOverflow)?;
                let next = used
                    .checked_add(len)
                    .filter(|next| *next <= payload.len())
                    .ok_or(ColumnError::LengthMismatch)?;
                let output_offset = if is_null {
                    None
                } else {
                    Some(
                        NonZeroUsize::new(used.checked_add(1).ok_or(ColumnError::LengthOverflow)?)
                            .unwrap(),
                    )
                };
                rows[(row - start) as usize + index] = RowSlot {
                    offset: output_offset,
                    len,
                };
                if matches!(self.spec.encoding, Encoding::Bytes { .. }) {
                    if len != 0 {
                        let source = self.blob + offset;
                        if run_len != 0 && run_source + run_len as u64 != source {
                            read_at(
                                self.spool,
                                run_source,
                                &mut payload[run_start..run_start + run_len],
                            )?;
                            run_len = 0;
                        }
                        if run_len == 0 {
                            run_source = source;
                            run_start = used;
                        }
                        run_len += len;
                    }
                } else {
                    payload[used..next].copy_from_slice(
                        &scratch[count + index * width..count + (index + 1) * width],
                    );
                }
                used = next;
            }
            if run_len != 0 {
                read_at(
                    self.spool,
                    run_source,
                    &mut payload[run_start..run_start + run_len],
                )?;
            }
            row += count as u32;
        }
        if used != payload.len() {
            return Err(ColumnError::LengthMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
