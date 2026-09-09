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

//! Bounded column payloads with borrowed readers and caller-owned output.
//!
//! Revision 1 header (64 bytes, little endian): V5CB[0..4], revision:u16[4..6],
//! header_size:u16[6..8], DataType:u8[8], encoding:u8[9], zero_flags[10..12],
//! column:u32[12..16], group:u64[16..24], row_start:u64[24..32],
//! row_count:u32[32..36], dictionary_count:u32[36..40], body_bytes:u64[40..48],
//! blob_bytes:u64[48..56], reserved_zero[56..64]. Encoding tags are AllNull=0,
//! Fixed=1, Variable=2, DictionaryText=3, TimestampParts=4. The body starts with ceil(rows/8)
//! null bytes, bit 1 meaning NULL, with zero unused high bits. Fixed values
//! use LE i64/f64 or 0/1 bytes. Variable values use rows+1 u32 offsets followed
//! by bytes. Dictionary text uses rows u32 IDs, count+1 u32 offsets, and UTF-8
//! bytes. Offsets start at zero, are nondecreasing, and end at blob length.
//! Timestamp Fixed values are i64 nanoseconds. TimestampParts uses 12 bytes per
//! row: i64 seconds followed by u32 subsecond nanoseconds, preserving Chrono's
//! full date range and leap seconds. NULL pairs are zero. DateTime inputs select
//! Fixed only when every non-NULL value round-trips exactly through i64 nanos.
//! Adapters must pass original timestamps before any lossy V4 nanos conversion.
//!
//! Blocks contain at most 4096 rows and 1 MiB decoded bytes, including the
//! header. Callers may lower either cap. V4 physical groups must be subdivided.
//! Page CRC and exact decompression precede parse; the page read plan must use
//! this decoded byte cap before reserving buffers. Expected group metadata
//! supplies the row cap before reading a compressed header. Full group/volume
//! coverage is the caller's responsibility. No codec operation allocates.
//!
//! JSON validates UTF-8, not JSON syntax. Vector bytes preserve IEEE values;
//! only byte shape and optional schema dimensions are validated. Dictionary
//! entries are strictly sorted and unique within this block. Existing V4
//! insertion-order dictionaries can use PlainTextFromDictionary without
//! walking unused dictionary entries or allocating a remapping table.

use std::fmt;

use chrono::{DateTime, Utc};

use crate::common::SmartString;
use crate::core::DataType;

pub const HEADER_BYTES: usize = 64;
pub const MAX_BLOCK_ROWS: u32 = 4096;
pub const MAX_DECODED_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnError {
    Length,
    Magic,
    Revision,
    Reserved,
    Type,
    Encoding,
    Identity,
    RowLimit,
    ByteLimit,
    CellLimit,
    DictionaryLimit,
    Overflow,
    Nulls,
    Boolean,
    Timestamp,
    Offsets,
    Utf8,
    DictionaryOrder,
    DictionaryId,
    VectorShape,
    OutputTooShort,
    UnsupportedEncoding,
}
impl fmt::Display for ColumnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid V5 column block: {self:?}")
    }
}
impl std::error::Error for ColumnError {}
type Result<T> = std::result::Result<T, ColumnError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnIdentity {
    pub physical_column: u32,
    pub group: u64,
    pub row_start: u64,
    pub row_count: u32,
    pub data_type: DataType,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnExpectation {
    pub identity: ColumnIdentity,
    /// None or zero means unspecified dimensions.
    pub vector_dimensions: Option<u16>,
}
impl ColumnExpectation {
    fn validate(self, limits: ColumnLimits) -> Result<()> {
        limits.validate(self.identity)?;
        if self.identity.data_type != DataType::Vector
            && self.vector_dimensions.is_some_and(|n| n != 0)
        {
            return Err(ColumnError::Type);
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnLimits {
    pub decoded_bytes: usize,
    pub rows: u32,
    pub dictionary_entries: u32,
    pub cell_bytes: usize,
}
impl Default for ColumnLimits {
    fn default() -> Self {
        Self {
            decoded_bytes: MAX_DECODED_BYTES,
            rows: MAX_BLOCK_ROWS,
            dictionary_entries: MAX_BLOCK_ROWS,
            cell_bytes: MAX_DECODED_BYTES,
        }
    }
}
impl ColumnLimits {
    fn validate(self, identity: ColumnIdentity) -> Result<()> {
        if self.rows == 0
            || self.rows > MAX_BLOCK_ROWS
            || identity.row_count == 0
            || identity.row_count > self.rows
        {
            return Err(ColumnError::RowLimit);
        }
        if !(HEADER_BYTES..=MAX_DECODED_BYTES).contains(&self.decoded_bytes) {
            return Err(ColumnError::ByteLimit);
        }
        if identity.physical_column == u32::MAX {
            return Err(ColumnError::Identity);
        }
        identity
            .row_start
            .checked_add(u64::from(identity.row_count))
            .ok_or(ColumnError::Overflow)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum ColumnInput<'a> {
    AllNull,
    I64(&'a [i64]),
    F64(&'a [f64]),
    TimestampNanos(&'a [i64]),
    /// Original timestamp values, before any V4 wrapping conversion. Chooses
    /// the common 8-byte representation or the lossless 12-byte representation.
    Timestamps(&'a [DateTime<Utc>]),
    Bool(&'a [bool]),
    /// Source ranges need not be contiguous; only selected non-null spans are
    /// read and repacked. DataType must be Text, Json, or Vector.
    Variable {
        data: &'a [u8],
        offsets: &'a [(u64, u64)],
    },
    PlainTextFromDictionary {
        ids: &'a [u32],
        dictionary: &'a [SmartString],
    },
    /// Requires caller-prepared sorted, unique, group-local dictionary and IDs.
    DictionaryText {
        ids: &'a [u32],
        dictionary: &'a [SmartString],
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Encoding {
    AllNull = 0,
    Fixed = 1,
    Variable = 2,
    DictionaryText = 3,
    TimestampParts = 4,
}
impl Encoding {
    fn decode(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::AllNull),
            1 => Ok(Self::Fixed),
            2 => Ok(Self::Variable),
            3 => Ok(Self::DictionaryText),
            4 => Ok(Self::TimestampParts),
            _ => Err(ColumnError::Encoding),
        }
    }
}

#[derive(Clone, Copy)]
struct Shape {
    expect: ColumnExpectation,
    encoding: Encoding,
    count: usize,
    dictionary_count: usize,
    null_bytes: usize,
    blob_bytes: usize,
    total: usize,
}
impl Shape {
    fn new(
        expect: ColumnExpectation,
        encoding: Encoding,
        dictionary_count: usize,
        blob_bytes: usize,
        limits: ColumnLimits,
    ) -> Result<Self> {
        expect.validate(limits)?;
        let count = expect.identity.row_count as usize;
        let null_bytes = count.div_ceil(8);
        let dt = expect.identity.data_type;
        let body = match encoding {
            Encoding::AllNull => {
                if dictionary_count != 0 || blob_bytes != 0 {
                    return Err(ColumnError::Length);
                }
                0
            }
            Encoding::Fixed => {
                if dictionary_count != 0 || blob_bytes != 0 {
                    return Err(ColumnError::Length);
                }
                count
                    .checked_mul(fixed_width(dt)?)
                    .ok_or(ColumnError::Overflow)?
            }
            Encoding::TimestampParts => {
                if dt != DataType::Timestamp {
                    return Err(ColumnError::Type);
                }
                if dictionary_count != 0 || blob_bytes != 0 {
                    return Err(ColumnError::Length);
                }
                count.checked_mul(12).ok_or(ColumnError::Overflow)?
            }
            Encoding::Variable => {
                if !matches!(dt, DataType::Text | DataType::Json | DataType::Vector) {
                    return Err(ColumnError::Type);
                }
                if dictionary_count != 0 {
                    return Err(ColumnError::Length);
                }
                offset_bytes(count)?
                    .checked_add(blob_bytes)
                    .ok_or(ColumnError::Overflow)?
            }
            Encoding::DictionaryText => {
                if dt != DataType::Text {
                    return Err(ColumnError::Type);
                }
                if dictionary_count > count || dictionary_count > limits.dictionary_entries as usize
                {
                    return Err(ColumnError::DictionaryLimit);
                }
                count
                    .checked_mul(4)
                    .and_then(|n| n.checked_add(offset_bytes(dictionary_count).ok()?))
                    .and_then(|n| n.checked_add(blob_bytes))
                    .ok_or(ColumnError::Overflow)?
            }
        };
        let total = HEADER_BYTES
            .checked_add(null_bytes)
            .and_then(|n| n.checked_add(body))
            .ok_or(ColumnError::Overflow)?;
        // Intra-page offsets are u32; the public cap is deliberately much lower.
        if total > limits.decoded_bytes || total > u32::MAX as usize || total > isize::MAX as usize
        {
            return Err(ColumnError::ByteLimit);
        }
        Ok(Self {
            expect,
            encoding,
            count,
            dictionary_count,
            null_bytes,
            blob_bytes,
            total,
        })
    }
}

pub struct ColumnEncodePlan<'a> {
    shape: Shape,
    nulls: &'a [bool],
    input: ColumnInput<'a>,
}
impl<'a> ColumnEncodePlan<'a> {
    /// Validate immutable inputs and exact output size before caller allocation.
    pub fn new(
        expect: ColumnExpectation,
        nulls: &'a [bool],
        input: ColumnInput<'a>,
        limits: ColumnLimits,
    ) -> Result<Self> {
        expect.validate(limits)?;
        let n = expect.identity.row_count as usize;
        if nulls.len() != n {
            return Err(ColumnError::Length);
        }
        let dt = expect.identity.data_type;
        let mut blob_bytes = 0usize;
        let mut dictionary_count = 0;
        let encoding = match input {
            ColumnInput::AllNull => {
                if nulls.iter().any(|&v| !v) {
                    return Err(ColumnError::Nulls);
                }
                Encoding::AllNull
            }
            ColumnInput::I64(v) => {
                validate_fixed(v.len(), n, dt, DataType::Integer)?;
                Encoding::Fixed
            }
            ColumnInput::F64(v) => {
                validate_fixed(v.len(), n, dt, DataType::Float)?;
                Encoding::Fixed
            }
            ColumnInput::TimestampNanos(v) => {
                validate_fixed(v.len(), n, dt, DataType::Timestamp)?;
                Encoding::Fixed
            }
            ColumnInput::Timestamps(values) => {
                validate_fixed(values.len(), n, dt, DataType::Timestamp)?;
                if values.iter().zip(nulls).all(|(value, &null)| {
                    null || value
                        .timestamp_nanos_opt()
                        .is_some_and(|nanos| DateTime::from_timestamp_nanos(nanos) == *value)
                }) {
                    Encoding::Fixed
                } else {
                    Encoding::TimestampParts
                }
            }
            ColumnInput::Bool(v) => {
                validate_fixed(v.len(), n, dt, DataType::Boolean)?;
                Encoding::Fixed
            }
            ColumnInput::Variable { data, offsets } => {
                if offsets.len() != n {
                    return Err(ColumnError::Length);
                }
                if !matches!(dt, DataType::Text | DataType::Json | DataType::Vector) {
                    return Err(ColumnError::Type);
                }
                for (i, &(offset, len)) in offsets.iter().enumerate() {
                    if nulls[i] {
                        continue;
                    }
                    let value = source_span(data, offset, len)?;
                    validate_value(value, expect, limits)?;
                    blob_bytes = add_blob(blob_bytes, value.len(), limits)?;
                }
                Encoding::Variable
            }
            ColumnInput::PlainTextFromDictionary { ids, dictionary } => {
                validate_fixed(ids.len(), n, dt, DataType::Text)?;
                for (i, &id) in ids.iter().enumerate() {
                    if nulls[i] {
                        continue;
                    }
                    let value = dictionary
                        .get(id as usize)
                        .ok_or(ColumnError::DictionaryId)?
                        .as_bytes();
                    validate_cell_length(value.len(), limits)?;
                    blob_bytes = add_blob(blob_bytes, value.len(), limits)?;
                }
                Encoding::Variable
            }
            ColumnInput::DictionaryText { ids, dictionary } => {
                validate_fixed(ids.len(), n, dt, DataType::Text)?;
                dictionary_count = dictionary.len();
                if dictionary_count > n || dictionary_count > limits.dictionary_entries as usize {
                    return Err(ColumnError::DictionaryLimit);
                }
                let mut previous: Option<&str> = None;
                for text in dictionary {
                    if previous.is_some_and(|p| p >= text.as_str()) {
                        return Err(ColumnError::DictionaryOrder);
                    }
                    validate_cell_length(text.len(), limits)?;
                    blob_bytes = add_blob(blob_bytes, text.len(), limits)?;
                    previous = Some(text.as_str());
                }
                for (i, &id) in ids.iter().enumerate() {
                    if !nulls[i] && id as usize >= dictionary_count {
                        return Err(ColumnError::DictionaryId);
                    }
                }
                Encoding::DictionaryText
            }
        };
        let shape = Shape::new(expect, encoding, dictionary_count, blob_bytes, limits)?;
        Ok(Self {
            shape,
            nulls,
            input,
        })
    }
    pub const fn encoded_len(&self) -> usize {
        self.shape.total
    }
    /// A short destination is unchanged. Successful writes leave its suffix
    /// unchanged; all other fallible work was completed by new().
    pub fn encode_into(&self, output: &mut [u8]) -> Result<usize> {
        let s = self.shape;
        if output.len() < s.total {
            return Err(ColumnError::OutputTooShort);
        }
        let output = &mut output[..s.total];
        output[..HEADER_BYTES + s.null_bytes].fill(0);
        output[..4].copy_from_slice(b"V5CB");
        put16(output, 4, 1);
        put16(output, 6, HEADER_BYTES as u16);
        output[8] = s.expect.identity.data_type as u8;
        output[9] = s.encoding as u8;
        put32(output, 12, s.expect.identity.physical_column);
        put64(output, 16, s.expect.identity.group);
        put64(output, 24, s.expect.identity.row_start);
        put32(output, 32, s.count as u32);
        put32(output, 36, s.dictionary_count as u32);
        put64(output, 40, (s.total - HEADER_BYTES) as u64);
        put64(output, 48, s.blob_bytes as u64);
        for (i, &null) in self.nulls.iter().enumerate() {
            if null {
                output[HEADER_BYTES + i / 8] |= 1 << (i % 8);
            }
        }
        let body = &mut output[HEADER_BYTES + s.null_bytes..];
        match self.input {
            ColumnInput::AllNull => (),
            ColumnInput::I64(v) | ColumnInput::TimestampNanos(v) => {
                for (i, bytes) in body.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    *bytes = if self.nulls[i] { 0 } else { v[i] }.to_le_bytes();
                }
            }
            ColumnInput::Timestamps(values) => {
                if s.encoding == Encoding::Fixed {
                    for (i, bytes) in body.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                        *bytes = if self.nulls[i] {
                            0
                        } else {
                            values[i]
                                .timestamp_nanos_opt()
                                .expect("validated nanos input")
                        }
                        .to_le_bytes();
                    }
                } else {
                    for (i, bytes) in body.as_chunks_mut::<12>().0.iter_mut().enumerate() {
                        let (seconds, nanos) = if self.nulls[i] {
                            (0, 0)
                        } else {
                            (values[i].timestamp(), values[i].timestamp_subsec_nanos())
                        };
                        bytes[..8].copy_from_slice(&seconds.to_le_bytes());
                        bytes[8..].copy_from_slice(&nanos.to_le_bytes());
                    }
                }
            }
            ColumnInput::F64(v) => {
                for (i, bytes) in body.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    *bytes = if self.nulls[i] { 0 } else { v[i].to_bits() }.to_le_bytes();
                }
            }
            ColumnInput::Bool(v) => {
                for (i, byte) in body.iter_mut().enumerate() {
                    *byte = u8::from(!self.nulls[i] && v[i]);
                }
            }
            ColumnInput::Variable { .. } | ColumnInput::PlainTextFromDictionary { .. } => {
                let (offsets, blob) = body.split_at_mut((s.count + 1) * 4);
                let mut position = 0;
                put32(offsets, 0, 0);
                for i in 0..s.count {
                    let value = if self.nulls[i] {
                        &[][..]
                    } else {
                        self.input_value(i)
                    };
                    blob[position..position + value.len()].copy_from_slice(value);
                    position += value.len();
                    put32(offsets, (i + 1) * 4, position as u32);
                }
            }
            ColumnInput::DictionaryText { ids, dictionary } => {
                let (id_bytes, rest) = body.split_at_mut(s.count * 4);
                for (i, bytes) in id_bytes.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    *bytes = if self.nulls[i] { 0 } else { ids[i] }.to_le_bytes();
                }
                let (offsets, blob) = rest.split_at_mut((s.dictionary_count + 1) * 4);
                let mut position = 0;
                put32(offsets, 0, 0);
                for (i, text) in dictionary.iter().enumerate() {
                    blob[position..position + text.len()].copy_from_slice(text.as_bytes());
                    position += text.len();
                    put32(offsets, (i + 1) * 4, position as u32);
                }
            }
        }
        Ok(s.total)
    }
    fn input_value(&self, i: usize) -> &'a [u8] {
        match self.input {
            ColumnInput::Variable { data, offsets } => {
                let (offset, len) = offsets[i];
                &data[offset as usize..(offset + len) as usize]
            }
            ColumnInput::PlainTextFromDictionary { ids, dictionary } => {
                dictionary[ids[i] as usize].as_bytes()
            }
            _ => unreachable!("validated variable encoding"),
        }
    }
}

/// Unaligned little-endian f32 payload. No Vec or native-alignment assumption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F32LeRef<'a> {
    bytes: &'a [u8],
}
impl<'a> F32LeRef<'a> {
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }
    pub const fn len(self) -> usize {
        self.bytes.len() / 4
    }
    pub const fn is_empty(self) -> bool {
        self.bytes.is_empty()
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = f32> + 'a {
        self.bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
    }
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColumnCell<'a> {
    Null(DataType),
    Integer(i64),
    Float(f64),
    TimestampNanos(i64),
    TimestampParts { seconds: i64, subsec_nanos: u32 },
    Boolean(bool),
    Text(&'a str),
    Json(&'a str),
    Vector(F32LeRef<'a>),
}

/// Fully validated immutable bytes. All returned references borrow this page,
/// never a reconstructed Value or dictionary allocation.
#[derive(Clone, Copy)]
pub struct ColumnBlockRef<'a> {
    shape: Shape,
    nulls: &'a [u8],
    body: &'a [u8],
    text_blob: Option<&'a str>,
}
/// Validated fixed-width LE lanes, independent of the input buffer's alignment.
#[derive(Clone, Copy)]
pub enum FixedValuesRef<'a> {
    I64(&'a [[u8; 8]]),
    F64(&'a [[u8; 8]]),
    TimestampNanos(&'a [[u8; 8]]),
    TimestampParts(TimestampPartsRef<'a>),
    Boolean(&'a [u8]),
}
/// Borrowed validated wide timestamp lanes, including zero pairs for NULL rows.
/// Compare (seconds, subsecond nanoseconds) directly for Chrono's ordering;
/// carrying a leap second's nanoseconds into the next second loses its identity.
#[derive(Clone, Copy, Debug)]
pub struct TimestampPartsRef<'a> {
    lanes: &'a [[u8; 12]],
}
impl<'a> TimestampPartsRef<'a> {
    pub const fn as_le_lanes(self) -> &'a [[u8; 12]] {
        self.lanes
    }
    pub const fn len(self) -> usize {
        self.lanes.len()
    }
    pub const fn is_empty(self) -> bool {
        self.lanes.is_empty()
    }
    pub fn get(self, local: usize) -> Option<(i64, u32)> {
        self.lanes.get(local).map(timestamp_parts)
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = (i64, u32)> + 'a {
        self.lanes.iter().map(timestamp_parts)
    }
}
fn timestamp_parts(bytes: &[u8; 12]) -> (i64, u32) {
    (
        i64::from_le_bytes(bytes[..8].try_into().expect("fixed seconds lane")),
        u32::from_le_bytes(bytes[8..].try_into().expect("fixed subsecond lane")),
    )
}
impl<'a> ColumnBlockRef<'a> {
    pub fn parse(bytes: &'a [u8], expect: ColumnExpectation, limits: ColumnLimits) -> Result<Self> {
        expect.validate(limits)?;
        if bytes.len() < HEADER_BYTES {
            return Err(ColumnError::Length);
        }
        if bytes.len() > limits.decoded_bytes {
            return Err(ColumnError::ByteLimit);
        }
        if &bytes[..4] != b"V5CB" {
            return Err(ColumnError::Magic);
        }
        if get16(bytes, 4) != 1 {
            return Err(ColumnError::Revision);
        }
        if get16(bytes, 6) != HEADER_BYTES as u16 {
            return Err(ColumnError::Length);
        }
        if bytes[10..12].iter().chain(&bytes[56..64]).any(|&v| v != 0) {
            return Err(ColumnError::Reserved);
        }
        let identity = ColumnIdentity {
            data_type: DataType::from_u8(bytes[8]).ok_or(ColumnError::Type)?,
            physical_column: get32(bytes, 12),
            group: get64(bytes, 16),
            row_start: get64(bytes, 24),
            row_count: get32(bytes, 32),
        };
        limits.validate(identity)?;
        if identity != expect.identity {
            return Err(ColumnError::Identity);
        }
        let encoding = Encoding::decode(bytes[9])?;
        let blob_bytes = usize::try_from(get64(bytes, 48)).map_err(|_| ColumnError::Overflow)?;
        let shape = Shape::new(
            expect,
            encoding,
            get32(bytes, 36) as usize,
            blob_bytes,
            limits,
        )?;
        if shape.total != bytes.len() || get64(bytes, 40) != (bytes.len() - HEADER_BYTES) as u64 {
            return Err(ColumnError::Length);
        }
        let nulls = &bytes[HEADER_BYTES..HEADER_BYTES + shape.null_bytes];
        let tail = shape.count % 8;
        if tail != 0 && nulls[shape.null_bytes - 1] & !((1u8 << tail) - 1) != 0 {
            return Err(ColumnError::Nulls);
        }
        let mut block = Self {
            shape,
            nulls,
            body: &bytes[HEADER_BYTES + shape.null_bytes..],
            text_blob: None,
        };
        match encoding {
            Encoding::AllNull => {
                if (0..shape.count).any(|i| !block.null_at(i)) {
                    return Err(ColumnError::Nulls);
                }
            }
            Encoding::Fixed => {
                if identity.data_type == DataType::Boolean && block.body.iter().any(|&b| b > 1) {
                    return Err(ColumnError::Boolean);
                }
            }
            Encoding::TimestampParts => {
                for (i, bytes) in block.body.as_chunks::<12>().0.iter().enumerate() {
                    let (seconds, nanos) = timestamp_parts(bytes);
                    if block.null_at(i) {
                        if seconds != 0 || nanos != 0 {
                            return Err(ColumnError::Nulls);
                        }
                    } else if DateTime::from_timestamp(seconds, nanos).is_none() {
                        return Err(ColumnError::Timestamp);
                    }
                }
            }
            Encoding::Variable => {
                let (offsets, blob) = block.body.split_at((shape.count + 1) * 4);
                validate_offsets(offsets, blob.len())?;
                if matches!(identity.data_type, DataType::Text | DataType::Json) {
                    block.text_blob =
                        Some(std::str::from_utf8(blob).map_err(|_| ColumnError::Utf8)?);
                }
                for i in 0..shape.count {
                    let value = value_span(offsets, blob, i);
                    if let Some(text) = block.text_blob {
                        // Whole-blob UTF-8 was checked once; offsets must still
                        // identify character boundaries for every individual cell.
                        checked_text_span(offsets, text, i)?;
                    }
                    if block.null_at(i) {
                        if !value.is_empty() {
                            return Err(ColumnError::Nulls);
                        }
                    } else if block.text_blob.is_some() {
                        validate_cell_length(value.len(), limits)?;
                    } else {
                        validate_value(value, expect, limits)?;
                    }
                }
            }
            Encoding::DictionaryText => {
                let (ids, rest) = block.body.split_at(shape.count * 4);
                let (offsets, blob) = rest.split_at((shape.dictionary_count + 1) * 4);
                validate_offsets(offsets, blob.len())?;
                let text_blob = std::str::from_utf8(blob).map_err(|_| ColumnError::Utf8)?;
                let mut previous = None;
                for i in 0..shape.dictionary_count {
                    let text = checked_text_span(offsets, text_blob, i)?;
                    validate_cell_length(text.len(), limits)?;
                    if previous.is_some_and(|p| p >= text) {
                        return Err(ColumnError::DictionaryOrder);
                    }
                    previous = Some(text);
                }
                for i in 0..shape.count {
                    if !block.null_at(i) && get32(ids, i * 4) as usize >= shape.dictionary_count {
                        return Err(ColumnError::DictionaryId);
                    }
                }
                block.text_blob = Some(text_blob);
            }
        }
        Ok(block)
    }
    pub const fn len(&self) -> usize {
        self.shape.count
    }
    pub const fn is_empty(&self) -> bool {
        self.shape.count == 0
    }
    pub const fn identity(&self) -> ColumnIdentity {
        self.shape.expect.identity
    }
    pub const fn null_bitmap(&self) -> &'a [u8] {
        self.nulls
    }
    pub fn fixed_values(&self) -> Option<FixedValuesRef<'a>> {
        if self.shape.encoding == Encoding::TimestampParts {
            return Some(FixedValuesRef::TimestampParts(TimestampPartsRef {
                lanes: self.body.as_chunks::<12>().0,
            }));
        }
        if self.shape.encoding != Encoding::Fixed {
            return None;
        }
        Some(match self.shape.expect.identity.data_type {
            DataType::Integer => FixedValuesRef::I64(self.body.as_chunks::<8>().0),
            DataType::Float => FixedValuesRef::F64(self.body.as_chunks::<8>().0),
            DataType::Timestamp => FixedValuesRef::TimestampNanos(self.body.as_chunks::<8>().0),
            DataType::Boolean => FixedValuesRef::Boolean(self.body),
            _ => unreachable!("validated fixed type"),
        })
    }
    pub fn is_null(&self, local: usize) -> Option<bool> {
        (local < self.len()).then(|| self.null_at(local))
    }
    fn null_at(&self, local: usize) -> bool {
        self.nulls[local / 8] & (1 << (local % 8)) != 0
    }
    pub fn cell(&self, local: usize) -> Option<ColumnCell<'a>> {
        if local >= self.len() {
            return None;
        }
        let dt = self.shape.expect.identity.data_type;
        if self.null_at(local) {
            return Some(ColumnCell::Null(dt));
        }
        let cell = match self.shape.encoding {
            Encoding::AllNull => unreachable!("validated all-null block"),
            Encoding::Fixed => match dt {
                DataType::Integer => ColumnCell::Integer(get64(self.body, local * 8) as i64),
                DataType::Timestamp => {
                    ColumnCell::TimestampNanos(get64(self.body, local * 8) as i64)
                }
                DataType::Float => ColumnCell::Float(f64::from_bits(get64(self.body, local * 8))),
                DataType::Boolean => ColumnCell::Boolean(self.body[local] != 0),
                _ => unreachable!("validated fixed type"),
            },
            Encoding::TimestampParts => {
                let (seconds, subsec_nanos) =
                    timestamp_parts(&self.body.as_chunks::<12>().0[local]);
                ColumnCell::TimestampParts {
                    seconds,
                    subsec_nanos,
                }
            }
            Encoding::Variable => {
                let (offsets, blob) = self.body.split_at((self.len() + 1) * 4);
                let bytes = value_span(offsets, blob, local);
                match dt {
                    DataType::Text => ColumnCell::Text(self.text_span(offsets, local)),
                    DataType::Json => ColumnCell::Json(self.text_span(offsets, local)),
                    DataType::Vector => ColumnCell::Vector(F32LeRef { bytes }),
                    _ => unreachable!("validated variable type"),
                }
            }
            Encoding::DictionaryText => {
                let id = get32(self.body, local * 4) as usize;
                ColumnCell::Text(self.dictionary_text(id))
            }
        };
        Some(cell)
    }
    pub fn cells(&self) -> impl ExactSizeIterator<Item = ColumnCell<'a>> + '_ {
        (0..self.len()).map(|i| self.cell(i).expect("bounded cell iteration"))
    }
    pub fn dictionary_id(&self, local: usize) -> Result<Option<Option<u32>>> {
        if self.shape.encoding != Encoding::DictionaryText {
            return Err(ColumnError::UnsupportedEncoding);
        }
        Ok((local < self.len()).then(|| {
            if self.null_at(local) {
                None
            } else {
                Some(get32(self.body, local * 4))
            }
        }))
    }
    pub fn dictionary_lookup(&self, text: &str) -> Result<Option<u32>> {
        if self.shape.encoding != Encoding::DictionaryText {
            return Err(ColumnError::UnsupportedEncoding);
        }
        let (mut left, mut right) = (0, self.shape.dictionary_count);
        while left < right {
            let mid = left + (right - left) / 2;
            match self.dictionary_text(mid).cmp(text) {
                std::cmp::Ordering::Less => left = mid + 1,
                std::cmp::Ordering::Greater => right = mid,
                std::cmp::Ordering::Equal => return Ok(Some(mid as u32)),
            }
        }
        Ok(None)
    }
    fn dictionary_text(&self, id: usize) -> &'a str {
        let rest = &self.body[self.len() * 4..];
        let offsets = &rest[..(self.shape.dictionary_count + 1) * 4];
        self.text_span(offsets, id)
    }
    fn text_span(&self, offsets: &[u8], id: usize) -> &'a str {
        let text = self.text_blob.expect("validated text backing");
        &text[get32(offsets, id * 4) as usize..get32(offsets, (id + 1) * 4) as usize]
    }
}

fn fixed_width(dt: DataType) -> Result<usize> {
    match dt {
        DataType::Integer | DataType::Float | DataType::Timestamp => Ok(8),
        DataType::Boolean => Ok(1),
        _ => Err(ColumnError::Type),
    }
}
fn validate_fixed(
    length: usize,
    expected: usize,
    actual: DataType,
    wanted: DataType,
) -> Result<()> {
    if actual != wanted {
        return Err(ColumnError::Type);
    }
    if length != expected {
        return Err(ColumnError::Length);
    }
    Ok(())
}
fn offset_bytes(count: usize) -> Result<usize> {
    count
        .checked_add(1)
        .and_then(|n| n.checked_mul(4))
        .ok_or(ColumnError::Overflow)
}
fn add_blob(current: usize, length: usize, limits: ColumnLimits) -> Result<usize> {
    let total = current.checked_add(length).ok_or(ColumnError::Overflow)?;
    if total > limits.decoded_bytes {
        return Err(ColumnError::ByteLimit);
    }
    Ok(total)
}
fn source_span(data: &[u8], offset: u64, len: u64) -> Result<&[u8]> {
    let end = offset.checked_add(len).ok_or(ColumnError::Overflow)?;
    let start = usize::try_from(offset).map_err(|_| ColumnError::Overflow)?;
    let end = usize::try_from(end).map_err(|_| ColumnError::Overflow)?;
    data.get(start..end).ok_or(ColumnError::Offsets)
}
fn validate_value(value: &[u8], expect: ColumnExpectation, limits: ColumnLimits) -> Result<()> {
    validate_cell_length(value.len(), limits)?;
    match expect.identity.data_type {
        DataType::Text | DataType::Json => {
            std::str::from_utf8(value).map_err(|_| ColumnError::Utf8)?;
        }
        DataType::Vector => {
            if !value.len().is_multiple_of(4) {
                return Err(ColumnError::VectorShape);
            }
            if let Some(dims) = expect.vector_dimensions.filter(|&n| n != 0) {
                let bytes = usize::from(dims)
                    .checked_mul(4)
                    .ok_or(ColumnError::Overflow)?;
                if value.len() != bytes {
                    return Err(ColumnError::VectorShape);
                }
            }
        }
        _ => return Err(ColumnError::Type),
    }
    Ok(())
}
fn validate_cell_length(length: usize, limits: ColumnLimits) -> Result<()> {
    if length > limits.cell_bytes {
        Err(ColumnError::CellLimit)
    } else if length > limits.decoded_bytes {
        // Reject before a UTF-8 scan even when the caller lowered only its
        // decoded page cap and left the per-cell limit at its default.
        Err(ColumnError::ByteLimit)
    } else {
        Ok(())
    }
}
fn checked_text_span<'a>(offsets: &[u8], text: &'a str, i: usize) -> Result<&'a str> {
    text.get(get32(offsets, i * 4) as usize..get32(offsets, (i + 1) * 4) as usize)
        .ok_or(ColumnError::Utf8)
}
fn validate_offsets(offsets: &[u8], blob_len: usize) -> Result<()> {
    let mut previous = 0;
    for (i, bytes) in offsets.as_chunks::<4>().0.iter().enumerate() {
        let value = u32::from_le_bytes(*bytes) as usize;
        if (i == 0 && value != 0) || value < previous || value > blob_len {
            return Err(ColumnError::Offsets);
        }
        previous = value;
    }
    if previous != blob_len {
        return Err(ColumnError::Offsets);
    }
    Ok(())
}
fn value_span<'a>(offsets: &[u8], blob: &'a [u8], i: usize) -> &'a [u8] {
    &blob[get32(offsets, i * 4) as usize..get32(offsets, (i + 1) * 4) as usize]
}
fn put16(b: &mut [u8], p: usize, v: u16) {
    b[p..p + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], p: usize, v: u32) {
    b[p..p + 4].copy_from_slice(&v.to_le_bytes());
}
fn put64(b: &mut [u8], p: usize, v: u64) {
    b[p..p + 8].copy_from_slice(&v.to_le_bytes());
}
fn get16(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes(b[p..p + 2].try_into().expect("checked column shape"))
}
fn get32(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().expect("checked column shape"))
}
fn get64(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().expect("checked column shape"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expected(dt: DataType, rows: u32) -> ColumnExpectation {
        ColumnExpectation {
            identity: ColumnIdentity {
                physical_column: 2,
                group: 3,
                row_start: 10,
                row_count: rows,
                data_type: dt,
            },
            vector_dimensions: None,
        }
    }
    fn encode(e: ColumnExpectation, nulls: &[bool], input: ColumnInput<'_>) -> Vec<u8> {
        let plan = ColumnEncodePlan::new(e, nulls, input, ColumnLimits::default()).unwrap();
        let n = plan.encoded_len();
        let mut bytes = vec![0xa5; n + 11];
        assert_eq!(plan.encode_into(&mut bytes).unwrap(), n);
        assert_eq!(&bytes[n..], &[0xa5; 11]);
        bytes.truncate(n);
        bytes
    }
    fn parse<'a>(bytes: &'a [u8], e: ColumnExpectation) -> Result<ColumnBlockRef<'a>> {
        ColumnBlockRef::parse(bytes, e, ColumnLimits::default())
    }

    #[test]
    fn column_timestamp_nanos_selection_is_lossless_and_ignores_null_values() {
        let nanos = [i64::MIN, -1, 0, i64::MAX];
        let values = nanos.map(DateTime::from_timestamp_nanos);
        let e = expected(DataType::Timestamp, 4);
        let bytes = encode(e, &[false; 4], ColumnInput::Timestamps(&values));
        assert_eq!(bytes[9], Encoding::Fixed as u8);
        let block = parse(&bytes, e).unwrap();
        for (i, n) in nanos.into_iter().enumerate() {
            assert_eq!(block.cell(i), Some(ColumnCell::TimestampNanos(n)));
        }

        let leap = DateTime::from_timestamp(1_483_228_799, 1_500_000_000).unwrap();
        // Some(nanos) alone is insufficient: it normalizes away leap identity.
        let flattened = DateTime::from_timestamp_nanos(leap.timestamp_nanos_opt().unwrap());
        assert_ne!(flattened, leap);
        assert!(leap < flattened);
        let mixed = [
            DateTime::<Utc>::MIN_UTC,
            values[1],
            leap,
            DateTime::<Utc>::MAX_UTC,
        ];
        let nulls = [true, false, true, true];
        let bytes = encode(e, &nulls, ColumnInput::Timestamps(&mixed));
        assert_eq!(bytes[9], Encoding::Fixed as u8);
        assert_eq!(
            parse(&bytes, e).unwrap().cell(1),
            Some(ColumnCell::TimestampNanos(-1))
        );
        let bytes = encode(e, &[true; 4], ColumnInput::Timestamps(&mixed));
        assert_eq!(bytes[9], Encoding::Fixed as u8);

        for timestamp in [
            values[0] - chrono::Duration::nanoseconds(1),
            values[3] + chrono::Duration::nanoseconds(1),
            leap,
        ] {
            let bytes = encode(
                expected(DataType::Timestamp, 1),
                &[false],
                ColumnInput::Timestamps(&[timestamp]),
            );
            assert_eq!(bytes[9], Encoding::TimestampParts as u8);
        }
        // Complete wide payload independently packed with Python struct/zlib.
        let bytes = encode(
            expected(DataType::Timestamp, 1),
            &[false],
            ColumnInput::Timestamps(&[leap]),
        );
        let hex = "5635434201004000050400000200000003000000000000000a0000000000000001000000000000000d0000000000000000000000000000000000000000000000007f46685800000000002f6859";
        let golden: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(bytes, golden);
        assert_eq!(crc32fast::hash(&bytes), 0x6b7f_06e4);
    }

    #[test]
    fn column_timestamp_parts_preserve_supported_value_domain_and_borrow_lanes() {
        let parsed = ["1500-01-01", "2500-01-01", "2016-12-31T23:59:60.5Z"]
            .map(|text| crate::core::value::parse_timestamp(text).unwrap());
        for (text, &timestamp) in ["1500-01-01", "2500-01-01", "2016-12-31T23:59:60.5Z"]
            .iter()
            .zip(&parsed)
        {
            assert_eq!(
                crate::core::Value::text(*text).coerce_to_type(DataType::Timestamp),
                crate::core::Value::timestamp(timestamp)
            );
        }
        let values = [
            DateTime::<Utc>::MIN_UTC,
            DateTime::<Utc>::MAX_UTC,
            // Chrono accepts a leap second even at the last supported date.
            DateTime::from_timestamp(DateTime::<Utc>::MAX_UTC.timestamp(), 1_999_999_999).unwrap(),
            parsed[0],
            parsed[1],
            parsed[2],
            DateTime::from_timestamp(-1, 1_500_000_000).unwrap(),
            DateTime::from_timestamp_nanos(-1),
        ];
        let e = expected(DataType::Timestamp, values.len() as u32);
        let bytes = encode(e, &[false; 8], ColumnInput::Timestamps(&values));
        assert_eq!(bytes[9], Encoding::TimestampParts as u8);
        assert_eq!(bytes.len(), HEADER_BYTES + 1 + values.len() * 12);
        let mut unaligned = vec![0xff];
        unaligned.extend_from_slice(&bytes);
        let block = parse(&unaligned[1..], e).unwrap();
        let FixedValuesRef::TimestampParts(lanes) = block.fixed_values().unwrap() else {
            panic!("wide timestamp lanes expected")
        };
        assert_eq!(lanes.len(), values.len());
        assert!(!lanes.is_empty());
        assert_eq!(lanes.get(values.len()), None);
        assert_eq!(
            lanes.as_le_lanes().as_ptr().cast::<u8>(),
            unaligned[1 + HEADER_BYTES + 1..].as_ptr()
        );
        for (i, ((seconds, subsec_nanos), value)) in lanes.iter().zip(values).enumerate() {
            assert_eq!(DateTime::from_timestamp(seconds, subsec_nanos), Some(value));
            assert_eq!(
                block.cell(i),
                Some(ColumnCell::TimestampParts {
                    seconds,
                    subsec_nanos
                })
            );
        }
        for left in values {
            for right in values {
                assert_eq!(
                    left.cmp(&right),
                    (left.timestamp(), left.timestamp_subsec_nanos())
                        .cmp(&(right.timestamp(), right.timestamp_subsec_nanos()))
                );
            }
        }
    }

    #[test]
    fn column_timestamp_parts_reject_invalid_pairs_type_shape_and_null_payloads() {
        let e = expected(DataType::Timestamp, 2);
        let values = [DateTime::<Utc>::MIN_UTC, DateTime::<Utc>::MAX_UTC];
        let original = encode(e, &[false, true], ColumnInput::Timestamps(&values));
        let body = HEADER_BYTES + 1;
        assert_eq!(&original[body + 12..], &[0; 12]);
        for (seconds, nanos) in [
            (i64::MIN, 0),
            (i64::MAX, 0),
            (0, 1_000_000_000),
            (59, 2_000_000_000),
            (-2, 1_500_000_000),
        ] {
            let mut corrupt = original.clone();
            put64(&mut corrupt, body, seconds as u64);
            put32(&mut corrupt, body + 8, nanos);
            assert!(matches!(parse(&corrupt, e), Err(ColumnError::Timestamp)));
        }
        let mut corrupt = original.clone();
        corrupt[body + 12] = 1;
        assert!(matches!(parse(&corrupt, e), Err(ColumnError::Nulls)));
        for field in [36, 48] {
            let mut corrupt = original.clone();
            corrupt[field] = 1;
            assert!(matches!(parse(&corrupt, e), Err(ColumnError::Length)));
        }
        let mut wrong_type = original.clone();
        wrong_type[8] = DataType::Integer as u8;
        assert!(matches!(
            parse(&wrong_type, expected(DataType::Integer, 2)),
            Err(ColumnError::Type)
        ));
        for len in 0..original.len() {
            assert!(parse(&original[..len], e).is_err());
        }
        let mut too_long = original.clone();
        too_long.push(0);
        assert!(matches!(parse(&too_long, e), Err(ColumnError::Length)));
    }

    #[test]
    fn column_timestamp_parts_obey_caps_before_output_mutation() {
        for rows in [4095, 4096] {
            let values = vec![DateTime::<Utc>::MAX_UTC; rows];
            let nulls = vec![false; rows];
            let e = expected(DataType::Timestamp, rows as u32);
            let plan = ColumnEncodePlan::new(
                e,
                &nulls,
                ColumnInput::Timestamps(&values),
                ColumnLimits::default(),
            )
            .unwrap();
            assert_eq!(
                plan.encoded_len(),
                HEADER_BYTES + rows.div_ceil(8) + rows * 12
            );
            let mut output = vec![0xa5; plan.encoded_len() - 1];
            assert_eq!(
                plan.encode_into(&mut output),
                Err(ColumnError::OutputTooShort)
            );
            assert!(output.iter().all(|&b| b == 0xa5));
            let limit = ColumnLimits {
                decoded_bytes: plan.encoded_len() - 1,
                ..ColumnLimits::default()
            };
            assert!(matches!(
                ColumnEncodePlan::new(e, &nulls, ColumnInput::Timestamps(&values), limit),
                Err(ColumnError::ByteLimit)
            ));
        }
        let values = [DateTime::<Utc>::MAX_UTC; 4097];
        assert!(matches!(
            ColumnEncodePlan::new(
                expected(DataType::Timestamp, 4097),
                &[false; 4097],
                ColumnInput::Timestamps(&values),
                ColumnLimits::default()
            ),
            Err(ColumnError::RowLimit)
        ));
    }

    #[test]
    fn column_golden_and_fixed_views_are_unaligned_and_bit_preserving() {
        let e = expected(DataType::Integer, 1);
        let bytes = encode(e, &[false], ColumnInput::I64(&[-9]));
        // Independently generated with Python struct.pack.
        let hex = "5635434201004000010100000200000003000000000000000a00000000000000010000000000000009000000000000000000000000000000000000000000000000f7ffffffffffffff";
        let golden: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        assert_eq!(bytes, golden);
        let mut unaligned = vec![0xff];
        unaligned.extend_from_slice(&bytes);
        let view = parse(&unaligned[1..], e).unwrap();
        assert_eq!(view.cell(0), Some(ColumnCell::Integer(-9)));
        assert_eq!(view.cell(1), None);
        match view.fixed_values().unwrap() {
            FixedValuesRef::I64(lanes) => assert_eq!(i64::from_le_bytes(lanes[0]), -9),
            _ => panic!("wrong fixed view"),
        }
        for dt in [DataType::Integer, DataType::Timestamp] {
            let e = expected(dt, 3);
            let values = [i64::MIN, -1, i64::MAX];
            let input = if dt == DataType::Integer {
                ColumnInput::I64(&values)
            } else {
                ColumnInput::TimestampNanos(&values)
            };
            let bytes = encode(e, &[false; 3], input);
            let got = parse(&bytes, e).unwrap();
            for (i, value) in values.into_iter().enumerate() {
                assert_eq!(
                    got.cell(i),
                    Some(if dt == DataType::Integer {
                        ColumnCell::Integer(value)
                    } else {
                        ColumnCell::TimestampNanos(value)
                    })
                );
            }
        }
        let bits = [
            0x8000_0000_0000_0000,
            0x7ff8_1234_5678_9abc,
            f64::INFINITY.to_bits(),
        ];
        let values = bits.map(f64::from_bits);
        let e = expected(DataType::Float, 3);
        let bytes = encode(e, &[false; 3], ColumnInput::F64(&values));
        let got = parse(&bytes, e).unwrap();
        for (cell, bits) in got.cells().zip(bits) {
            let ColumnCell::Float(value) = cell else {
                panic!("float expected")
            };
            assert_eq!(value.to_bits(), bits);
        }
    }

    #[test]
    fn column_all_encoding_wire_images_match_independent_goldens() {
        let dictionary = [SmartString::from("z")];
        // Complete header/body CRCs independently packed by Python struct/zlib.
        // The first golden test additionally compares every byte of an I64 page.
        let cases = [
            (DataType::Json, ColumnInput::AllNull, true, 65, 0x05fc_ba9a),
            (
                DataType::Float,
                ColumnInput::F64(&[-0.0]),
                false,
                73,
                0x82fe_4bd8,
            ),
            (
                DataType::Timestamp,
                ColumnInput::TimestampNanos(&[-1]),
                false,
                73,
                0x1f55_78bd,
            ),
            (
                DataType::Boolean,
                ColumnInput::Bool(&[true]),
                false,
                66,
                0xd3b2_4800,
            ),
            (
                DataType::Text,
                ColumnInput::Variable {
                    data: "é".as_bytes(),
                    offsets: &[(0, 2)],
                },
                false,
                75,
                0xc438_6e6e,
            ),
            (
                DataType::Json,
                ColumnInput::Variable {
                    data: b"no",
                    offsets: &[(0, 2)],
                },
                false,
                75,
                0x7843_b4b4,
            ),
            (
                DataType::Vector,
                ColumnInput::Variable {
                    data: &[0, 0, 0, 0x80],
                    offsets: &[(0, 4)],
                },
                false,
                77,
                0x4e7f_223a,
            ),
            (
                DataType::Text,
                ColumnInput::DictionaryText {
                    ids: &[0],
                    dictionary: &dictionary,
                },
                false,
                78,
                0xb541_41f8,
            ),
        ];
        for (dt, input, null, length, crc) in cases {
            let bytes = encode(expected(dt, 1), &[null], input);
            assert_eq!(bytes.len(), length);
            assert_eq!(crc32fast::hash(&bytes), crc, "{dt:?}");
        }
    }

    #[test]
    fn column_nulls_booleans_and_row_limits_are_checked() {
        for n in [1, 7, 8, 9, 4095, 4096] {
            let e = expected(DataType::Boolean, n);
            let nulls: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
            let values: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
            let bytes = encode(e, &nulls, ColumnInput::Bool(&values));
            let view = parse(&bytes, e).unwrap();
            for i in 0..n as usize {
                assert_eq!(
                    view.cell(i),
                    Some(if nulls[i] {
                        ColumnCell::Null(DataType::Boolean)
                    } else {
                        ColumnCell::Boolean(values[i])
                    })
                );
            }
            if n % 8 != 0 {
                let mut bad = bytes.clone();
                bad[64 + n as usize / 8] |= 0x80;
                assert!(matches!(parse(&bad, e), Err(ColumnError::Nulls)));
            }
            let mut bad = bytes;
            bad[64 + (n as usize).div_ceil(8)] = 2;
            assert!(matches!(parse(&bad, e), Err(ColumnError::Boolean)));
        }
        for dt in [
            DataType::Null,
            DataType::Integer,
            DataType::Float,
            DataType::Text,
            DataType::Boolean,
            DataType::Timestamp,
            DataType::Json,
            DataType::Vector,
        ] {
            let e = expected(dt, 9);
            let mut bytes = encode(e, &[true; 9], ColumnInput::AllNull);
            assert!(parse(&bytes, e)
                .unwrap()
                .cells()
                .all(|v| v == ColumnCell::Null(dt)));
            bytes[64] &= !1;
            assert!(matches!(parse(&bytes, e), Err(ColumnError::Nulls)));
        }
        for n in [0, 4097] {
            assert!(matches!(
                ColumnEncodePlan::new(
                    expected(DataType::Integer, n),
                    &[],
                    ColumnInput::AllNull,
                    ColumnLimits::default()
                ),
                Err(ColumnError::RowLimit)
            ));
        }
        assert!(matches!(
            ColumnEncodePlan::new(
                expected(DataType::Integer, 2),
                &[true; 2],
                ColumnInput::AllNull,
                ColumnLimits {
                    rows: 1,
                    ..ColumnLimits::default()
                }
            ),
            Err(ColumnError::RowLimit)
        ));
    }

    #[test]
    fn column_variable_json_and_vectors_keep_shape_and_borrow_storage() {
        for dt in [DataType::Text, DataType::Json] {
            let e = expected(dt, 4);
            let data = "é\0not json".as_bytes();
            let spans = [(0, 2), (2, 0), (2, 9), (u64::MAX, u64::MAX)];
            let bytes = encode(
                e,
                &[false, false, false, true],
                ColumnInput::Variable {
                    data,
                    offsets: &spans,
                },
            );
            let view = parse(&bytes, e).unwrap();
            for (i, expected) in ["é", "", "\0not json"].into_iter().enumerate() {
                let text = match view.cell(i).unwrap() {
                    ColumnCell::Text(v) | ColumnCell::Json(v) => v,
                    _ => panic!(),
                };
                assert_eq!(text, expected);
                let p = text.as_ptr() as usize;
                assert!(p >= bytes.as_ptr() as usize && p <= bytes.as_ptr() as usize + bytes.len());
            }
            assert_eq!(view.cell(3), Some(ColumnCell::Null(dt)));
        }
        let bits = [0x8000_0000u32, 0x7fc1_2345, f32::INFINITY.to_bits()];
        let packed: Vec<u8> = bits.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut e = expected(DataType::Vector, 1);
        e.vector_dimensions = Some(3);
        let bytes = encode(
            e,
            &[false],
            ColumnInput::Variable {
                data: &packed,
                offsets: &[(0, 12)],
            },
        );
        let ColumnCell::Vector(vector) = parse(&bytes, e).unwrap().cell(0).unwrap() else {
            panic!()
        };
        assert_eq!(vector.iter().map(f32::to_bits).collect::<Vec<_>>(), bits);
        e.vector_dimensions = Some(2);
        assert!(matches!(parse(&bytes, e), Err(ColumnError::VectorShape)));
        e.vector_dimensions = None;
        assert!(matches!(
            ColumnEncodePlan::new(
                e,
                &[false],
                ColumnInput::Variable {
                    data: &packed,
                    offsets: &[(0, 11)]
                },
                ColumnLimits::default()
            ),
            Err(ColumnError::VectorShape)
        ));
        let bytes = encode(
            e,
            &[false],
            ColumnInput::Variable {
                data: &[],
                offsets: &[(0, 0)],
            },
        );
        assert!(
            matches!(parse(&bytes, e).unwrap().cell(0), Some(ColumnCell::Vector(v)) if v.is_empty())
        );
    }

    #[test]
    fn column_dictionary_is_sorted_local_and_plain_source_may_be_unsorted() {
        let dict: Vec<SmartString> = ["", "a", "é"].into_iter().map(SmartString::from).collect();
        let e = expected(DataType::Text, 4);
        let bytes = encode(
            e,
            &[false, false, false, true],
            ColumnInput::DictionaryText {
                ids: &[2, 0, 1, u32::MAX],
                dictionary: &dict,
            },
        );
        let view = parse(&bytes, e).unwrap();
        assert_eq!(view.dictionary_lookup("é").unwrap(), Some(2));
        assert_eq!(view.dictionary_lookup("missing").unwrap(), None);
        assert_eq!(view.dictionary_id(3).unwrap(), Some(None));
        assert_eq!(view.dictionary_id(4).unwrap(), None);
        assert_eq!(view.cell(0), Some(ColumnCell::Text("é")));
        for bad in [["z", "a"], ["a", "a"]] {
            let bad = bad.map(SmartString::from);
            assert!(matches!(
                ColumnEncodePlan::new(
                    expected(DataType::Text, 2),
                    &[false; 2],
                    ColumnInput::DictionaryText {
                        ids: &[0, 1],
                        dictionary: &bad
                    },
                    ColumnLimits::default()
                ),
                Err(ColumnError::DictionaryOrder)
            ));
            let plain = encode(
                expected(DataType::Text, 2),
                &[false; 2],
                ColumnInput::PlainTextFromDictionary {
                    ids: &[0, 1],
                    dictionary: &bad,
                },
            );
            let view = parse(&plain, expected(DataType::Text, 2)).unwrap();
            assert_eq!(view.cell(0), Some(ColumnCell::Text(bad[0].as_str())));
            assert!(matches!(
                view.dictionary_lookup("a"),
                Err(ColumnError::UnsupportedEncoding)
            ));
        }
        let e = expected(DataType::Text, 1);
        let bytes = encode(
            e,
            &[true],
            ColumnInput::DictionaryText {
                ids: &[u32::MAX],
                dictionary: &[],
            },
        );
        assert_eq!(
            parse(&bytes, e).unwrap().cell(0),
            Some(ColumnCell::Null(DataType::Text))
        );
        assert!(matches!(
            ColumnEncodePlan::new(
                e,
                &[false],
                ColumnInput::DictionaryText {
                    ids: &[0],
                    dictionary: &[]
                },
                ColumnLimits::default()
            ),
            Err(ColumnError::DictionaryId)
        ));
    }

    #[test]
    fn column_rejects_corruption_before_exposing_borrowed_values() {
        let e = expected(DataType::Text, 2);
        let bytes = encode(
            e,
            &[false; 2],
            ColumnInput::Variable {
                data: "éz".as_bytes(),
                offsets: &[(0, 2), (2, 1)],
            },
        );
        for n in 0..bytes.len() {
            assert!(parse(&bytes[..n], e).is_err(), "truncation {n}");
        }
        let mut bad = bytes.clone();
        bad.push(0);
        assert!(parse(&bad, e).is_err());
        for (position, value) in [
            (0, 0),
            (4, 2),
            (6, 63),
            (8, 255),
            (9, 255),
            (10, 1),
            (12, 3),
            (16, 4),
            (24, 11),
            (32, 3),
            (36, 1),
            (40, 1),
            (48, 99),
            (56, 1),
        ] {
            let mut bad = bytes.clone();
            bad[position] = value;
            assert!(parse(&bad, e).is_err(), "byte {position}");
        }
        for offsets in [[1, 2, 3], [0, 3, 2], [0, 2, 4], [0, 1, 3]] {
            let mut bad = bytes.clone();
            for (i, value) in offsets.into_iter().enumerate() {
                put32(&mut bad, 65 + i * 4, value);
            }
            assert!(parse(&bad, e).is_err());
        }
        let mut bad = bytes.clone();
        bad[64] = 1;
        assert!(matches!(parse(&bad, e), Err(ColumnError::Nulls)));
        let mut bad = bytes;
        bad[77] = 255;
        assert!(matches!(parse(&bad, e), Err(ColumnError::Utf8)));
        let dict = [SmartString::from("a"), SmartString::from("z")];
        let bytes = encode(
            e,
            &[false; 2],
            ColumnInput::DictionaryText {
                ids: &[0, 1],
                dictionary: &dict,
            },
        );
        let mut bad = bytes.clone();
        put32(&mut bad, 65, 2);
        assert!(matches!(parse(&bad, e), Err(ColumnError::DictionaryId)));
        let mut bad = bytes;
        bad[85] = b'z';
        bad[86] = b'a';
        assert!(matches!(parse(&bad, e), Err(ColumnError::DictionaryOrder)));
    }

    #[test]
    fn column_preflight_caps_and_invalid_ranges_never_mutate_output() {
        let e = expected(DataType::Integer, 1);
        let plan =
            ColumnEncodePlan::new(e, &[false], ColumnInput::I64(&[1]), ColumnLimits::default())
                .unwrap();
        let mut out = [0xa5; 72];
        assert_eq!(plan.encode_into(&mut out), Err(ColumnError::OutputTooShort));
        assert_eq!(out, [0xa5; 72]);
        assert!(matches!(
            ColumnEncodePlan::new(
                e,
                &[false],
                ColumnInput::I64(&[1]),
                ColumnLimits {
                    decoded_bytes: 72,
                    ..ColumnLimits::default()
                }
            ),
            Err(ColumnError::ByteLimit)
        ));
        let mut overflow = e;
        overflow.identity.row_start = u64::MAX;
        assert!(matches!(
            ColumnEncodePlan::new(
                overflow,
                &[false],
                ColumnInput::I64(&[1]),
                ColumnLimits::default()
            ),
            Err(ColumnError::Overflow)
        ));
        for span in [(1, 9), (u64::MAX, 1)] {
            assert!(ColumnEncodePlan::new(
                expected(DataType::Text, 1),
                &[false],
                ColumnInput::Variable {
                    data: b"abc",
                    offsets: &[span]
                },
                ColumnLimits::default()
            )
            .is_err());
        }
        let e = expected(DataType::Text, 1);
        // Invalid UTF-8 must not be scanned beyond the active decoded cap.
        assert!(matches!(
            ColumnEncodePlan::new(
                e,
                &[false],
                ColumnInput::Variable {
                    data: &[0xff; 129],
                    offsets: &[(0, 129)]
                },
                ColumnLimits {
                    decoded_bytes: 128,
                    ..ColumnLimits::default()
                }
            ),
            Err(ColumnError::ByteLimit)
        ));
        assert!(matches!(
            ColumnEncodePlan::new(
                e,
                &[false],
                ColumnInput::Variable {
                    data: b"abc",
                    offsets: &[(0, 3)]
                },
                ColumnLimits {
                    cell_bytes: 2,
                    ..ColumnLimits::default()
                }
            ),
            Err(ColumnError::CellLimit)
        ));
        let huge = vec![b'a'; MAX_DECODED_BYTES];
        assert!(matches!(
            ColumnEncodePlan::new(
                e,
                &[false],
                ColumnInput::Variable {
                    data: &huge,
                    offsets: &[(0, huge.len() as u64)]
                },
                ColumnLimits::default()
            ),
            Err(ColumnError::ByteLimit)
        ));
        assert!(matches!(
            ColumnEncodePlan::new(
                e,
                &[false],
                ColumnInput::Variable {
                    data: b"a",
                    offsets: &[(0, 1)]
                },
                ColumnLimits {
                    decoded_bytes: MAX_DECODED_BYTES + 1,
                    ..ColumnLimits::default()
                }
            ),
            Err(ColumnError::ByteLimit)
        ));
    }

    #[test]
    fn column_dimensions_are_bound_to_vector_type_and_dictionary_boundaries_are_utf8() {
        let mut e = expected(DataType::Text, 2);
        let dictionary = [SmartString::from("é"), SmartString::from("ê")];
        let bytes = encode(
            e,
            &[false; 2],
            ColumnInput::DictionaryText {
                ids: &[0, 1],
                dictionary: &dictionary,
            },
        );
        let mut malformed = bytes.clone();
        // UTF-8 blob is valid as a whole, but an offset splits its first scalar.
        put32(&mut malformed, 65 + 8 + 4, 1);
        assert!(matches!(parse(&malformed, e), Err(ColumnError::Utf8)));
        e.vector_dimensions = Some(2);
        assert!(matches!(parse(&bytes, e), Err(ColumnError::Type)));
        assert!(matches!(
            ColumnEncodePlan::new(e, &[true; 2], ColumnInput::AllNull, ColumnLimits::default()),
            Err(ColumnError::Type)
        ));
    }

    #[test]
    fn column_page_io_valid_checksum_does_not_hide_invalid_typed_payload() {
        use super::super::directory::{
            DirectoryKey, DirectoryRoot, Layout, LeafEntry, RootSummary, RowBounds, Section,
            KEY_REQUIRED,
        };
        use super::super::envelope::{Codec, FileIdentity, Header, ReadLimits};
        use super::super::page_io::{OpenedEnvelope, PageReadPlan, PageWriter, ReadAt};
        struct Source<'a>(&'a [u8]);
        impl ReadAt for Source<'_> {
            fn read_at(&self, offset: u64, out: &mut [u8]) -> std::io::Result<usize> {
                let Some(bytes) = usize::try_from(offset).ok().and_then(|p| self.0.get(p..)) else {
                    return Ok(0);
                };
                let n = out.len().min(bytes.len()).min(13);
                out[..n].copy_from_slice(&bytes[..n]);
                Ok(n)
            }
        }
        let e = ColumnExpectation {
            identity: ColumnIdentity {
                physical_column: 0,
                group: 0,
                row_start: 0,
                row_count: 2,
                data_type: DataType::Integer,
            },
            vector_dimensions: None,
        };
        let limits = ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: MAX_DECODED_BYTES as u64,
            page_decoded_bytes: MAX_DECODED_BYTES as u64,
        };
        for corrupt in [false, true] {
            let mut payload = encode(e, &[false; 2], ColumnInput::I64(&[12, 34]));
            if corrupt {
                payload[56] = 1;
            }
            let compressed = lz4_flex::block::compress(&payload);
            let mut file = Vec::new();
            let identity = FileIdentity::new(1, 2, 3).unwrap();
            let mut writer = PageWriter::new(&mut file, Header::new(identity), limits).unwrap();
            let page = writer
                .append_stored(Codec::Lz4Block, &compressed, payload.len() as u64)
                .unwrap();
            let leaf = writer
                .append_leaf(
                    &[LeafEntry {
                        key: DirectoryKey {
                            section: Section::ColumnBlocks as u16,
                            flags: KEY_REQUIRED,
                            column: 0,
                            ordinal: 0,
                        },
                        page,
                    }],
                    &mut [0; 80],
                )
                .unwrap();
            let finished = writer
                .finish(&RootSummary {
                    layout: Layout::RowId,
                    legacy_base: None,
                    row_count: 2,
                    column_count: 1,
                    group_count: 1,
                    entry_count: 1,
                    rows: Some(RowBounds { min: 0, max: 1 }),
                    window: None,
                    directory: Some(DirectoryRoot {
                        depth: 1,
                        page: leaf,
                    }),
                })
                .unwrap();
            let source = Source(&file);
            OpenedEnvelope::read(&source, finished.footer.file_length, &limits)
                .unwrap()
                .require_identity(identity)
                .unwrap();
            let read = PageReadPlan::for_page(&finished.footer, page, &limits).unwrap();
            let mut stored = vec![0; read.stored_buffer_len()];
            let mut decoded = vec![0; read.decoded_buffer_len()];
            let raw = read.read_into(&source, &mut stored, &mut decoded).unwrap();
            if corrupt {
                assert!(matches!(parse(raw, e), Err(ColumnError::Reserved)));
            } else {
                assert_eq!(
                    parse(raw, e).unwrap().cell(1),
                    Some(ColumnCell::Integer(34))
                );
            }
        }
    }
}
