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

//! Ephemeral captured-row backing for a single inactive V5 build.
//!
//! The caller supplies exclusive streams, immutable captured values and schema,
//! reservations, and every working buffer. No method opens or removes files,
//! clones rows/values, obtains an arena lock, or establishes durable provenance.
//! Call these operations AFTER releasing capture/publication locks. A finished
//! spool remains bound to the same borrowed schema, file identity, schema
//! version and source context; its backing must remain immutable, including
//! through aliases. Checksums detect corruption, not adversarial substitution.
//!
//! Each row has a 64-byte checksummed identity header, one 32-byte cell descriptor
//! per physical column, and contiguous payload bytes. The header binds signed
//! row ID, creator, explicit DML/base source, exact frame length, schema binding
//! and descriptor CRC. Each descriptor binds column/type/null, byte range and
//! cell CRC. Selected-column gathers validate ALL descriptor metadata and use a
//! caller-owned read window with bounded adjacent-byte prefetch. Unselected payload CRCs are checked when those
//! columns are gathered. Timestamp pairs retain seconds/nanoseconds, including
//! leap seconds; floating/vector bits and empty-versus-NULL remain unchanged.
//!
//! This is not an installed file format. Opaque summaries delimit logical bytes;
//! stale tails in reused scratch files are ignored. Row positions must travel
//! with the exact finished spool and external sort. Duplicate row IDs are errors,
//! not a choice of precedence. The capture coordinator chooses authoritative rows.
//!
//! Gather capacity errors leave caller output unchanged (metadata scratch can
//! change). I/O/corruption or panic permanently poisons a reader/writer; partial
//! gather output is invalid. Large captured cells can remain serialized intact
//! above V5's page cap: emission returns OversizedCell until a future large-value
//! policy handles them. No value is dropped, shortened or assigned a fake source.

use std::fmt;
use std::io::{self, Seek, SeekFrom, Write};
use std::num::NonZeroU64;

use chrono::{DateTime, Utc};

use crate::core::{DataType, Value};

use super::column_block::{ColumnInput, MAX_BLOCK_ROWS, MAX_DECODED_BYTES};
use super::envelope::FileIdentity;
use super::page_io::ReadAt;
use super::payload_writer::ColumnSpec;
use super::row_identity::{RowSource, SourceEncodingContext};

pub mod runs;

const HEADER_BYTES: usize = 64;
const CELL_BYTES: usize = 32;
pub const MAX_IO_BYTES: usize = 64 * 1024;
pub const MAX_ROW_BYTES: u64 = 256 * 1024 * 1024;
pub const MAX_SPOOL_BYTES: u64 = 1 << 40;

#[derive(Debug)]
pub enum SpoolError {
    Io(io::Error),
    Limits,
    Overflow,
    Identity,
    Source,
    Schema,
    Type,
    Length,
    Tag,
    Checksum,
    Utf8,
    Timestamp,
    VectorShape,
    BufferTooSmall,
    OversizedCell,
    UnexpectedEof,
    InvalidIoCount,
    RowOrder,
    NotSorted,
    AlreadySorted,
    Poisoned,
    Finished,
}
impl fmt::Display for SpoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V5 captured row spool: {self:?}")
    }
}
impl std::error::Error for SpoolError {}
impl From<io::Error> for SpoolError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
type Result<T> = std::result::Result<T, SpoolError>;

#[derive(Clone, Copy, Debug)]
pub struct SpoolLimits {
    pub rows: u64,
    pub spool_bytes: u64,
    pub row_bytes: u64,
    pub cell_bytes: u64,
    pub columns: u32,
}
impl Default for SpoolLimits {
    fn default() -> Self {
        Self {
            rows: 1 << 32,
            spool_bytes: MAX_SPOOL_BYTES,
            row_bytes: MAX_ROW_BYTES,
            cell_bytes: MAX_ROW_BYTES,
            columns: 65_535,
        }
    }
}
impl SpoolLimits {
    fn validate(self) -> Result<()> {
        if self.rows == 0
            || self.rows > MAX_SPOOL_BYTES / HEADER_BYTES as u64
            || !(HEADER_BYTES as u64..=MAX_SPOOL_BYTES).contains(&self.spool_bytes)
            || !(HEADER_BYTES as u64..=MAX_ROW_BYTES).contains(&self.row_bytes)
            || self.cell_bytes > self.row_bytes
            || self.columns > 65_535
        {
            return Err(SpoolError::Limits);
        }
        Ok(())
    }
}

/// Immutable captured metadata; no inferred source or schema conversion.
#[derive(Clone, Copy)]
pub struct SpoolBinding<'a> {
    pub identity: FileIdentity,
    pub schema_version: NonZeroU64,
    pub columns: &'a [ColumnSpec],
    pub source: SourceEncodingContext,
}
impl SpoolBinding<'_> {
    fn fingerprint(self) -> u32 {
        let mut crc = crc32fast::Hasher::new();
        for value in [
            self.identity.table_id.get(),
            self.identity.incarnation.get(),
            self.identity.volume_id.get(),
            self.schema_version.get(),
        ] {
            crc.update(&value.to_le_bytes());
        }
        crc.update(&(self.columns.len() as u64).to_le_bytes());
        for column in self.columns {
            crc.update(&[column.data_type as u8]);
            crc.update(&column.vector_dimensions.unwrap_or(0).to_le_bytes());
        }
        let base = self.source.legacy_base();
        crc.update(&[u8::from(base.is_some())]);
        crc.update(&base.map_or(0, |base| base.generation.get()).to_le_bytes());
        crc.update(&base.map_or(0, |base| base.barrier_lsn).to_le_bytes());
        crc.finalize()
    }
    fn source_lane(self, source: RowSource) -> Result<u64> {
        match source {
            RowSource::LegacyBase if self.source.legacy_base().is_some() => Ok(0),
            RowSource::Dml(lsn)
                if self
                    .source
                    .legacy_base()
                    .is_none_or(|base| lsn.get() > base.barrier_lsn) =>
            {
                Ok(lsn.get())
            }
            _ => Err(SpoolError::Source),
        }
    }
}

pub struct CapturedRow<'a> {
    pub row_id: i64,
    pub creator_txn_id: i64,
    pub source: RowSource,
    pub values: &'a [Value],
}

/// One indivisible external-sort record. EMPTY initializes caller scratch only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowPosition {
    row_id: i64,
    creator: i64,
    source: u64,
    offset: u64,
    len: u64,
    header_crc: u32,
}
impl RowPosition {
    pub const EMPTY: Self = Self {
        row_id: 0,
        creator: 0,
        source: 0,
        offset: 0,
        len: 0,
        header_crc: 0,
    };
    pub const fn row_id(self) -> i64 {
        self.row_id
    }
    pub const fn creator_txn_id(self) -> i64 {
        self.creator
    }
    pub fn source(self) -> RowSource {
        NonZeroU64::new(self.source).map_or(RowSource::LegacyBase, RowSource::Dml)
    }
}

#[derive(Clone, Copy)]
pub struct FinishedSpool<'a> {
    binding: SpoolBinding<'a>,
    fingerprint: u32,
    limits: SpoolLimits,
    rows: u64,
    bytes: u64,
}
impl<'schema> FinishedSpool<'schema> {
    pub const fn row_count(self) -> u64 {
        self.rows
    }
    pub const fn byte_len(self) -> u64 {
        self.bytes
    }
    pub const fn binding(self) -> SpoolBinding<'schema> {
        self.binding
    }
    fn validate_position(self, row: RowPosition) -> Result<()> {
        if row.creator == 0 || row.creator < -1 || row.len < HEADER_BYTES as u64 {
            return Err(SpoolError::Identity);
        }
        if row.len > self.limits.row_bytes
            || row
                .offset
                .checked_add(row.len)
                .is_none_or(|end| end > self.bytes)
        {
            return Err(SpoolError::Length);
        }
        self.binding.source_lane(row.source())?;
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Open,
    Poisoned,
    Finished,
}
impl State {
    fn check(self) -> Result<()> {
        match self {
            Self::Open => Ok(()),
            Self::Poisoned => Err(SpoolError::Poisoned),
            Self::Finished => Err(SpoolError::Finished),
        }
    }
}

pub struct SpoolWriter<'a, 'schema, W: Write + ?Sized> {
    sink: &'a mut W,
    buffer: &'a mut [u8],
    buffered: usize,
    summary: FinishedSpool<'schema>,
    state: State,
}
impl<'a, 'schema, W: Write + Seek + ?Sized> SpoolWriter<'a, 'schema, W> {
    pub fn new(
        sink: &'a mut W,
        binding: SpoolBinding<'schema>,
        limits: SpoolLimits,
        buffer: &'a mut [u8],
    ) -> Result<Self> {
        limits.validate()?;
        check_io(buffer)?;
        if binding.columns.len() > limits.columns as usize {
            return Err(SpoolError::Schema);
        }
        for column in binding.columns {
            if column.data_type != DataType::Vector
                && column.vector_dimensions.is_some_and(|n| n != 0)
            {
                return Err(SpoolError::Schema);
            }
        }
        if sink.seek(SeekFrom::Start(0))? != 0 {
            return Err(SpoolError::Identity);
        }
        Ok(Self {
            sink,
            buffer,
            buffered: 0,
            summary: FinishedSpool {
                binding,
                fingerprint: binding.fingerprint(),
                limits,
                rows: 0,
                bytes: 0,
            },
            state: State::Open,
        })
    }
}
impl<'schema, W: Write + ?Sized> SpoolWriter<'_, 'schema, W> {
    /// Accepts one immutable row into the caller buffer/sink. The receipt is
    /// provisional until finish flushes all buffered rows successfully; a later
    /// I/O error invalidates the entire unfinished build, including receipts.
    /// Semantic/type/size preflight does not mutate scratch or the sink.
    pub fn append(&mut self, row: CapturedRow<'_>) -> Result<RowPosition> {
        self.state.check()?;
        let binding = self.summary.binding;
        if row.values.len() != binding.columns.len() {
            return Err(SpoolError::Schema);
        }
        if row.creator_txn_id == 0 || row.creator_txn_id < -1 {
            return Err(SpoolError::Identity);
        }
        let source = binding.source_lane(row.source)?;
        if self.summary.rows == self.summary.limits.rows {
            return Err(SpoolError::Limits);
        }
        let body_start = HEADER_BYTES as u64 + row.values.len() as u64 * CELL_BYTES as u64;
        let mut end = body_start;
        let mut descriptors = crc32fast::Hasher::new();
        for (ordinal, (value, column)) in row.values.iter().zip(binding.columns).enumerate() {
            let bytes = value_bytes(value, *column)?;
            if bytes.as_slice().len() as u64 > self.summary.limits.cell_bytes {
                return Err(SpoolError::OversizedCell);
            }
            let descriptor = encode_cell(
                ordinal as u32,
                *column,
                value.is_null(),
                end,
                bytes.as_slice(),
            );
            descriptors.update(&descriptor);
            end = end
                .checked_add(bytes.as_slice().len() as u64)
                .ok_or(SpoolError::Overflow)?;
        }
        let next = self
            .summary
            .bytes
            .checked_add(end)
            .ok_or(SpoolError::Overflow)?;
        if end > self.summary.limits.row_bytes || next > self.summary.limits.spool_bytes {
            return Err(SpoolError::Limits);
        }
        let mut header = [0; HEADER_BYTES];
        header[..4].copy_from_slice(b"V5RS");
        header[4..6].copy_from_slice(&1u16.to_le_bytes());
        header[6..8].copy_from_slice(&(HEADER_BYTES as u16).to_le_bytes());
        header[8..16].copy_from_slice(&end.to_le_bytes());
        header[16..24].copy_from_slice(&row.row_id.to_le_bytes());
        header[24..32].copy_from_slice(&row.creator_txn_id.to_le_bytes());
        header[32..40].copy_from_slice(&source.to_le_bytes());
        header[40..44].copy_from_slice(&(row.values.len() as u32).to_le_bytes());
        header[44..48].copy_from_slice(&self.summary.fingerprint.to_le_bytes());
        header[48..52].copy_from_slice(&descriptors.finalize().to_le_bytes());
        let checksum = crc32fast::hash(&header[..60]);
        header[60..].copy_from_slice(&checksum.to_le_bytes());
        let position = RowPosition {
            row_id: row.row_id,
            creator: row.creator_txn_id,
            source,
            offset: self.summary.bytes,
            len: end,
            header_crc: checksum,
        };
        self.state = State::Poisoned;
        self.write(&header)?;
        let mut at = body_start;
        for (ordinal, (value, column)) in row.values.iter().zip(binding.columns).enumerate() {
            let bytes = value_bytes(value, *column)?;
            self.write(&encode_cell(
                ordinal as u32,
                *column,
                value.is_null(),
                at,
                bytes.as_slice(),
            ))?;
            at += bytes.as_slice().len() as u64;
        }
        for (value, column) in row.values.iter().zip(binding.columns) {
            self.write(value_bytes(value, *column)?.as_slice())?;
        }
        self.summary.bytes = next;
        self.summary.rows += 1;
        self.state = State::Open;
        Ok(position)
    }
    pub fn finish(&mut self) -> Result<FinishedSpool<'schema>> {
        self.state.check()?;
        self.state = State::Poisoned;
        self.flush_buffer()?;
        self.state = State::Finished;
        Ok(self.summary)
    }
    fn write(&mut self, mut bytes: &[u8]) -> Result<()> {
        while !bytes.is_empty() {
            if self.buffered == self.buffer.len() {
                self.flush_buffer()?;
            }
            let count = bytes.len().min(self.buffer.len() - self.buffered);
            self.buffer[self.buffered..self.buffered + count].copy_from_slice(&bytes[..count]);
            self.buffered += count;
            bytes = &bytes[count..];
        }
        Ok(())
    }
    fn flush_buffer(&mut self) -> Result<()> {
        write_all(self.sink, &self.buffer[..self.buffered])?;
        self.buffered = 0;
        Ok(())
    }
}

enum ValueBytes<'a> {
    Borrowed(&'a [u8]),
    Inline([u8; 12], usize),
}
impl ValueBytes<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Inline(bytes, len) => &bytes[..*len],
        }
    }
}
fn value_bytes(value: &Value, column: ColumnSpec) -> Result<ValueBytes<'_>> {
    let mut scalar = [0; 12];
    let width = match value {
        Value::Null(_) => return Ok(ValueBytes::Borrowed(&[])),
        Value::Integer(value) if column.data_type == DataType::Integer => {
            scalar[..8].copy_from_slice(&value.to_le_bytes());
            8
        }
        Value::Float(value) if column.data_type == DataType::Float => {
            scalar[..8].copy_from_slice(&value.to_bits().to_le_bytes());
            8
        }
        Value::Boolean(value) if column.data_type == DataType::Boolean => {
            scalar[0] = u8::from(*value);
            1
        }
        Value::Timestamp(value) if column.data_type == DataType::Timestamp => {
            scalar[..8].copy_from_slice(&value.timestamp().to_le_bytes());
            scalar[8..].copy_from_slice(&value.timestamp_subsec_nanos().to_le_bytes());
            12
        }
        Value::Text(value) if column.data_type == DataType::Text => {
            return Ok(ValueBytes::Borrowed(value.as_bytes()))
        }
        Value::Extension(value) if value.first() == Some(&(column.data_type as u8)) => {
            let bytes = &value[1..];
            match column.data_type {
                DataType::Json => {
                    std::str::from_utf8(bytes).map_err(|_| SpoolError::Utf8)?;
                }
                DataType::Vector => validate_vector(bytes.len() as u64, column)?,
                _ => return Err(SpoolError::Type),
            }
            return Ok(ValueBytes::Borrowed(bytes));
        }
        _ => return Err(SpoolError::Type),
    };
    Ok(ValueBytes::Inline(scalar, width))
}
fn validate_vector(len: u64, column: ColumnSpec) -> Result<()> {
    if !len.is_multiple_of(4)
        || column
            .vector_dimensions
            .is_some_and(|n| n != 0 && len != u64::from(n) * 4)
    {
        return Err(SpoolError::VectorShape);
    }
    Ok(())
}
fn encode_cell(
    ordinal: u32,
    column: ColumnSpec,
    null: bool,
    offset: u64,
    bytes: &[u8],
) -> [u8; CELL_BYTES] {
    let mut output = [0; CELL_BYTES];
    output[..8].copy_from_slice(&offset.to_le_bytes());
    output[8..16].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    output[16..20].copy_from_slice(&crc32fast::hash(bytes).to_le_bytes());
    output[20..24].copy_from_slice(&ordinal.to_le_bytes());
    output[24] = column.data_type as u8;
    output[25] = u8::from(null);
    let checksum = crc32fast::hash(&output[..28]);
    output[28..].copy_from_slice(&checksum.to_le_bytes());
    output
}

/// Caller-owned metadata scratch for one selected column, never all columns.
#[derive(Clone, Copy)]
pub struct CellSpan {
    offset: u64,
    len: u64,
    checksum: u32,
    null: bool,
}
impl CellSpan {
    pub const EMPTY: Self = Self {
        offset: 0,
        len: 0,
        checksum: 0,
        null: true,
    };
}

/// Each slice is a reservation supplied by the caller; no variant allocates.
pub enum ColumnBuffer<'a> {
    AllNull,
    I64(&'a mut [i64]),
    F64(&'a mut [f64]),
    Bool(&'a mut [bool]),
    Timestamps(&'a mut [DateTime<Utc>]),
    Variable {
        bytes: &'a mut [u8],
        offsets: &'a mut [(u64, u64)],
    },
}
pub struct GatheredColumn<'a> {
    pub nulls: &'a [bool],
    pub input: ColumnInput<'a>,
}

/// A descriptor-validated group bound to this reader and immutable receipts.
/// The borrow prevents swapping backing/schema/positions between validation and
/// column gathers. Only one selected-column span slice is needed, not an array
/// for every column. Caller backing must remain immutable through all aliases.
pub struct PreparedGroup<'group, 'io, 'schema, R: ReadAt + ?Sized> {
    reader: &'group mut SpoolReader<'io, 'schema, R>,
    rows: &'group [RowPosition],
}
impl<R: ReadAt + ?Sized> PreparedGroup<'_, '_, '_, R> {
    pub fn gather_column<'out>(
        &mut self,
        column: u32,
        spans: &mut [CellSpan],
        nulls: &'out mut [bool],
        output: ColumnBuffer<'out>,
    ) -> Result<GatheredColumn<'out>> {
        self.reader
            .gather_inner(self.rows, column, spans, nulls, output, true)
    }
}

pub struct SpoolReader<'a, 'schema, R: ReadAt + ?Sized> {
    window: ReadWindow<'a, R>,
    summary: FinishedSpool<'schema>,
    state: State,
}
impl<'a, 'schema, R: ReadAt + ?Sized> SpoolReader<'a, 'schema, R> {
    pub fn new(
        source: &'a R,
        summary: FinishedSpool<'schema>,
        buffer: &'a mut [u8],
    ) -> Result<Self> {
        check_io(buffer)?;
        Ok(Self {
            window: ReadWindow {
                source,
                buffer,
                offset: 0,
                len: 0,
                end: summary.bytes,
            },
            summary,
            state: State::Open,
        })
    }
    /// Validate every row frame/descriptor exactly once before transposition.
    /// Each following column gather validates only its selected descriptor and
    /// cell CRC, with the same capacity/poison guarantees as the one-shot API.
    pub fn prepare_group<'group>(
        &'group mut self,
        rows: &'group [RowPosition],
    ) -> Result<PreparedGroup<'group, 'a, 'schema, R>> {
        self.state.check()?;
        if rows.is_empty() || rows.len() > MAX_BLOCK_ROWS as usize {
            return Err(SpoolError::Limits);
        }
        for &row in rows {
            self.summary.validate_position(row)?;
        }
        self.state = State::Poisoned;
        for &row in rows {
            self.inspect(row, 0)?;
        }
        self.state = State::Open;
        Ok(PreparedGroup { reader: self, rows })
    }
    /// One-shot convenience for a single column. Use prepare_group when
    /// transposing multiple columns to avoid repeated full-descriptor walks.
    pub fn gather_column<'out>(
        &mut self,
        rows: &[RowPosition],
        column: u32,
        spans: &mut [CellSpan],
        nulls: &'out mut [bool],
        output: ColumnBuffer<'out>,
    ) -> Result<GatheredColumn<'out>> {
        self.gather_inner(rows, column, spans, nulls, output, false)
    }
    fn gather_inner<'out>(
        &mut self,
        rows: &[RowPosition],
        column: u32,
        spans: &mut [CellSpan],
        nulls: &'out mut [bool],
        output: ColumnBuffer<'out>,
        prepared: bool,
    ) -> Result<GatheredColumn<'out>> {
        self.state.check()?;
        let spec = *self
            .summary
            .binding
            .columns
            .get(column as usize)
            .ok_or(SpoolError::Schema)?;
        if rows.is_empty() || rows.len() > MAX_BLOCK_ROWS as usize {
            return Err(SpoolError::Limits);
        }
        if spans.len() < rows.len() || nulls.len() < rows.len() {
            return Err(SpoolError::BufferTooSmall);
        }
        match &output {
            ColumnBuffer::AllNull if spec.data_type == DataType::Null => (),
            ColumnBuffer::I64(values)
                if spec.data_type == DataType::Integer && values.len() >= rows.len() => {}
            ColumnBuffer::F64(values)
                if spec.data_type == DataType::Float && values.len() >= rows.len() => {}
            ColumnBuffer::Bool(values)
                if spec.data_type == DataType::Boolean && values.len() >= rows.len() => {}
            ColumnBuffer::Timestamps(values)
                if spec.data_type == DataType::Timestamp && values.len() >= rows.len() => {}
            ColumnBuffer::Variable { offsets, .. }
                if matches!(
                    spec.data_type,
                    DataType::Text | DataType::Json | DataType::Vector
                ) && offsets.len() >= rows.len() => {}
            _ => return Err(SpoolError::BufferTooSmall),
        }
        for row in rows {
            self.summary.validate_position(*row)?;
        }
        self.state = State::Poisoned;
        let mut required = 0u64;
        for (row, span) in rows.iter().zip(spans.iter_mut()) {
            *span = if prepared {
                self.inspect_selected(*row, column)?
            } else {
                self.inspect(*row, column)?
            };
            required = required.checked_add(span.len).ok_or(SpoolError::Overflow)?;
        }
        // These semantic caps do not invalidate otherwise valid spool backing.
        if spans[..rows.len()]
            .iter()
            .any(|span| span.len > MAX_DECODED_BYTES as u64)
        {
            self.state = State::Open;
            return Err(SpoolError::OversizedCell);
        }
        if let ColumnBuffer::Variable { bytes, .. } = &output {
            if required > bytes.len() as u64 {
                self.state = State::Open;
                return Err(SpoolError::BufferTooSmall);
            }
        }
        let spans = &spans[..rows.len()];
        let nulls = &mut nulls[..rows.len()];
        let input = match output {
            ColumnBuffer::AllNull => ColumnInput::AllNull,
            ColumnBuffer::I64(values) => {
                for (value, span) in values.iter_mut().zip(spans) {
                    let mut lane = [0; 8];
                    self.read_cell(*span, &mut lane)?;
                    *value = i64::from_le_bytes(lane);
                }
                ColumnInput::I64(&values[..rows.len()])
            }
            ColumnBuffer::F64(values) => {
                for (value, span) in values.iter_mut().zip(spans) {
                    let mut lane = [0; 8];
                    self.read_cell(*span, &mut lane)?;
                    *value = f64::from_bits(u64::from_le_bytes(lane));
                }
                ColumnInput::F64(&values[..rows.len()])
            }
            ColumnBuffer::Bool(values) => {
                for (value, span) in values.iter_mut().zip(spans) {
                    let mut lane = [0];
                    self.read_cell(*span, &mut lane)?;
                    if lane[0] > 1 {
                        return Err(SpoolError::Tag);
                    }
                    *value = lane[0] != 0;
                }
                ColumnInput::Bool(&values[..rows.len()])
            }
            ColumnBuffer::Timestamps(values) => {
                for (value, span) in values.iter_mut().zip(spans) {
                    let mut lane = [0; 12];
                    self.read_cell(*span, &mut lane)?;
                    *value = DateTime::from_timestamp(i64_at(&lane, 0), u32_at(&lane, 8))
                        .ok_or(SpoolError::Timestamp)?;
                }
                ColumnInput::Timestamps(&values[..rows.len()])
            }
            ColumnBuffer::Variable { bytes, offsets } => {
                let mut at = 0usize;
                for (offset, span) in offsets.iter_mut().zip(spans) {
                    let len = usize::try_from(span.len).map_err(|_| SpoolError::Overflow)?;
                    self.read_cell(*span, &mut bytes[at..at + len])?;
                    if matches!(spec.data_type, DataType::Text | DataType::Json) {
                        std::str::from_utf8(&bytes[at..at + len]).map_err(|_| SpoolError::Utf8)?;
                    }
                    *offset = (at as u64, len as u64);
                    at += len;
                }
                ColumnInput::Variable {
                    data: &bytes[..at],
                    offsets: &offsets[..rows.len()],
                }
            }
        };
        for (null, span) in nulls.iter_mut().zip(spans) {
            *null = span.null;
        }
        self.state = State::Open;
        Ok(GatheredColumn { nulls, input })
    }
    fn inspect(&mut self, row: RowPosition, selected: u32) -> Result<CellSpan> {
        let mut header = [0; HEADER_BYTES];
        self.window.read(row.offset, &mut header)?;
        if crc32fast::hash(&header[..60]) != u32_at(&header, 60)
            || u32_at(&header, 60) != row.header_crc
        {
            return Err(SpoolError::Checksum);
        }
        if &header[..4] != b"V5RS" || header[4..8] != [1, 0, 64, 0] || header[52..60] != [0; 8] {
            return Err(SpoolError::Tag);
        }
        if u64_at(&header, 8) != row.len
            || i64_at(&header, 16) != row.row_id
            || i64_at(&header, 24) != row.creator
            || u64_at(&header, 32) != row.source
            || u32_at(&header, 40) as usize != self.summary.binding.columns.len()
            || u32_at(&header, 44) != self.summary.fingerprint
        {
            return Err(SpoolError::Identity);
        }
        let columns = self.summary.binding.columns;
        let mut end = HEADER_BYTES as u64 + columns.len() as u64 * CELL_BYTES as u64;
        if end > row.len {
            return Err(SpoolError::Length);
        }
        let mut checksum = crc32fast::Hasher::new();
        let mut found = CellSpan::EMPTY;
        for ordinal in 0..columns.len() {
            let mut entry = [0; CELL_BYTES];
            self.window.read(
                row.offset + HEADER_BYTES as u64 + (ordinal * CELL_BYTES) as u64,
                &mut entry,
            )?;
            checksum.update(&entry);
            let span = self.parse_cell(&entry, row, ordinal)?;
            if span.offset - row.offset != end {
                return Err(SpoolError::Length);
            }
            if ordinal == selected as usize {
                found = span;
            }
            end = end.checked_add(span.len).ok_or(SpoolError::Overflow)?;
        }
        if end != row.len || checksum.finalize() != u32_at(&header, 48) {
            return Err(SpoolError::Checksum);
        }
        Ok(found)
    }
    fn inspect_selected(&mut self, row: RowPosition, selected: u32) -> Result<CellSpan> {
        let mut entry = [0; CELL_BYTES];
        self.window.read(
            row.offset + HEADER_BYTES as u64 + u64::from(selected) * CELL_BYTES as u64,
            &mut entry,
        )?;
        self.parse_cell(&entry, row, selected as usize)
    }
    fn parse_cell(
        &self,
        entry: &[u8; CELL_BYTES],
        row: RowPosition,
        ordinal: usize,
    ) -> Result<CellSpan> {
        let columns = self.summary.binding.columns;
        let spec = columns[ordinal];
        if crc32fast::hash(&entry[..28]) != u32_at(entry, 28) {
            return Err(SpoolError::Checksum);
        }
        if u32_at(entry, 20) as usize != ordinal
            || entry[24] != spec.data_type as u8
            || entry[25] > 1
            || entry[26..28] != [0, 0]
        {
            return Err(SpoolError::Tag);
        }
        let offset = u64_at(entry, 0);
        let len = u64_at(entry, 8);
        let null = entry[25] != 0;
        if offset < HEADER_BYTES as u64 + columns.len() as u64 * CELL_BYTES as u64
            || offset.checked_add(len).is_none_or(|end| end > row.len)
            || len > self.summary.limits.cell_bytes
            || (null && (len != 0 || u32_at(entry, 16) != crc32fast::hash(&[])))
        {
            return Err(SpoolError::Length);
        }
        if !null {
            let fixed = match spec.data_type {
                DataType::Integer | DataType::Float => Some(8),
                DataType::Boolean => Some(1),
                DataType::Timestamp => Some(12),
                DataType::Null => return Err(SpoolError::Type),
                DataType::Vector => {
                    validate_vector(len, spec)?;
                    None
                }
                _ => None,
            };
            if fixed.is_some_and(|expected| len != expected) {
                return Err(SpoolError::Length);
            }
        }
        Ok(CellSpan {
            offset: row.offset + offset,
            len,
            checksum: u32_at(entry, 16),
            null,
        })
    }
    fn read_cell(&mut self, span: CellSpan, output: &mut [u8]) -> Result<()> {
        if span.null {
            output.fill(0);
            return Ok(());
        }
        if output.len() as u64 != span.len {
            return Err(SpoolError::Length);
        }
        self.window.read(span.offset, output)?;
        if crc32fast::hash(output) != span.checksum {
            return Err(SpoolError::Checksum);
        }
        Ok(())
    }
}

/// Exclusive caller scratch. Logical end bounds every prefetch, even on reused
/// backing with a stale tail. Arbitrarily sorted row positions can still seek.
struct ReadWindow<'a, R: ReadAt + ?Sized> {
    source: &'a R,
    buffer: &'a mut [u8],
    offset: u64,
    len: usize,
    end: u64,
}
impl<R: ReadAt + ?Sized> ReadWindow<'_, R> {
    fn read(&mut self, mut offset: u64, mut output: &mut [u8]) -> Result<()> {
        if offset
            .checked_add(output.len() as u64)
            .is_none_or(|end| end > self.end)
        {
            return Err(SpoolError::Length);
        }
        while !output.is_empty() {
            if offset < self.offset || offset - self.offset >= self.len as u64 {
                // Stable windows reuse adjacent metadata in either row order.
                // Starting every refill at a descending row would reread a full
                // window per row, even when those rows occupy the same page.
                let start = offset / self.buffer.len() as u64 * self.buffer.len() as u64;
                let len = (self.end - start).min(self.buffer.len() as u64) as usize;
                self.len = 0;
                read_exact(self.source, start, &mut self.buffer[..len])?;
                self.offset = start;
                self.len = len;
            }
            let local = (offset - self.offset) as usize;
            let count = output.len().min(self.len - local);
            output[..count].copy_from_slice(&self.buffer[local..local + count]);
            output = &mut output[count..];
            offset += count as u64;
        }
        Ok(())
    }
}

fn check_io(bytes: &[u8]) -> Result<()> {
    if !(HEADER_BYTES..=MAX_IO_BYTES).contains(&bytes.len()) {
        return Err(SpoolError::BufferTooSmall);
    }
    Ok(())
}
fn read_exact<R: ReadAt + ?Sized>(source: &R, mut offset: u64, mut bytes: &mut [u8]) -> Result<()> {
    while !bytes.is_empty() {
        match source.read_at(offset, bytes) {
            Ok(0) => return Err(SpoolError::UnexpectedEof),
            Ok(n) if n > bytes.len() => return Err(SpoolError::InvalidIoCount),
            Ok(n) => {
                offset = offset.checked_add(n as u64).ok_or(SpoolError::Overflow)?;
                bytes = &mut bytes[n..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => (),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}
fn write_all<W: Write + ?Sized>(sink: &mut W, mut bytes: &[u8]) -> Result<()> {
    while !bytes.is_empty() {
        match sink.write(bytes) {
            Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero).into()),
            Ok(n) if n > bytes.len() => return Err(SpoolError::InvalidIoCount),
            Ok(n) => bytes = &bytes[n..],
            Err(error) if error.kind() == io::ErrorKind::Interrupted => (),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}
fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}
fn i64_at(bytes: &[u8], at: usize) -> i64 {
    i64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests;
