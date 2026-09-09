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

//! Inactive, bounded sorted-spool planning and V5 payload emission.
//!
//! Planning writes only a caller-owned checksummed group-boundary stream. Its
//! private FinishedPlan binds the exact sorted receipts, immutable schema/source
//! context and configuration. No V5 byte is emitted until the complete plan has
//! succeeded. Emission reuses one column's typed scratch and the existing
//! PayloadWriter; the caller then finishes its external descriptor directory.
//!
//! Group bytes mean the SUM of decoded column pages plus the actual group
//! metadata page (64 bytes), row-ID page (32+8*n) and source page (32+8*n).
//! Directory pages are separately reserved. Plain variable encoding is explicit;
//! dictionary inputs are never silently converted. Initial emission uses Raw
//! pages and requires their full decoded sizes to fit stored-page limits too.
//! Candidate sizing scans each bounded batch once: exact per-row payload/lane
//! costs and rounded group overhead, with conservative 12-byte timestamp lanes.
//! Final groups are encoded-sized exactly. Timestamp groups may be smaller than
//! maximal packing. A column-page cap below the group cap conservatively also
//! caps whole-group candidates (the defaults coincide). An exact one-row fallback
//! never rejects a fitting row merely because of either bound. Each column is
//! gathered once per group.
//!
//! All streams, reservations, file ownership and immutable-backing guarantees
//! belong to the caller. Scratch may change on failure. A failed/unwinding pass
//! returns no publishable capability and its destination is invalid; restarting
//! means rewriting the entire pass, not resuming a partial prefix. This module
//! does not open, delete, sync or install files, hold arena locks, certify durable
//! bootstrap, or enable the engine's memory limit. Existing oversized rows fail
//! planning explicitly and remain in their original spool/hot source.

use std::fmt;
use std::io::{Seek, SeekFrom, Write};

use chrono::{DateTime, Utc};

use super::column_block::{
    ColumnEncodePlan, ColumnError, ColumnExpectation, ColumnIdentity, ColumnLimits, MAX_BLOCK_ROWS,
    MAX_DECODED_BYTES,
};
use super::directory::{Layout, RowBounds, VolumeShape, ROOT_SUMMARY_BYTES};
use super::directory_writer::ENCODING_BYTES;
use super::envelope::{Header, ReadLimits};
use super::group_metadata::GroupRecord;
use super::page_io::ReadAt;
use super::payload_writer::{
    ColumnSpec, CompletedPayloads, DescriptorSink, PayloadError, PayloadWriter,
};
use super::row_identity::{PlannedCheckpoint, RowSource, SourceEncodingContext};
use super::row_spool::runs::{RowRuns, SortedRows};
use super::row_spool::{
    read_exact, write_all, CellSpan, ColumnBuffer, FinishedSpool, GatheredColumn, PreparedGroup,
    RowPosition, SpoolError, SpoolReader, MAX_IO_BYTES,
};
use crate::core::DataType;

const PLAN_RECORD_BYTES: usize = 64;
const GROUP_FIXED_BYTES: u64 = 128;
const GROUP_LANE_BYTES: u64 = 16;

#[derive(Debug)]
pub enum CoordinatorError {
    Spool(SpoolError),
    Column(ColumnError),
    Payload(PayloadError),
    Configuration,
    BufferTooSmall,
    Overflow,
    Identity,
    PlanChecksum,
    PlanRecord,
    Coverage,
    CannotFitRow { row_id: i64, decoded_limit: usize },
    MinimumGroupBytes { required: u64, limit: usize },
}
impl fmt::Display for CoordinatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V5 coordinator: {self:?}")
    }
}
impl std::error::Error for CoordinatorError {}
impl From<SpoolError> for CoordinatorError {
    fn from(v: SpoolError) -> Self {
        Self::Spool(v)
    }
}
impl From<ColumnError> for CoordinatorError {
    fn from(v: ColumnError) -> Self {
        Self::Column(v)
    }
}
impl From<PayloadError> for CoordinatorError {
    fn from(v: PayloadError) -> Self {
        Self::Payload(v)
    }
}
type Result<T> = std::result::Result<T, CoordinatorError>;

#[derive(Clone, Copy, Debug)]
pub struct BuildConfig {
    pub rows: u32,
    pub group_decoded_bytes: usize,
    pub columns: ColumnLimits,
    pub pages: ReadLimits,
}
impl BuildConfig {
    fn validate(self, spool: FinishedSpool<'_>) -> Result<usize> {
        if self.rows == 0
            || self.rows > MAX_BLOCK_ROWS
            || self.columns.rows < self.rows
            || self.columns.rows > MAX_BLOCK_ROWS
            || !(144..=MAX_DECODED_BYTES).contains(&self.group_decoded_bytes)
            || !(64..=MAX_DECODED_BYTES).contains(&self.columns.decoded_bytes)
            || self.pages.root_stored_bytes < ROOT_SUMMARY_BYTES as u64
            || self.pages.root_decoded_bytes < ROOT_SUMMARY_BYTES as u64
            || self.pages.page_stored_bytes < ENCODING_BYTES as u64
            || self.pages.page_decoded_bytes < ENCODING_BYTES as u64
        {
            return Err(CoordinatorError::Configuration);
        }
        let page_cap = self
            .pages
            .page_stored_bytes
            .min(self.pages.page_decoded_bytes);
        if self.columns.decoded_bytes as u64 > page_cap {
            return Err(CoordinatorError::Configuration);
        }
        let row_cap = u64::from(self.rows)
            .min((page_cap - 32) / 8)
            .min(spool.row_count()) as usize;
        if spool.row_count() != 0 {
            let minimum = (spool.binding().columns.len() as u64)
                .checked_mul(65)
                .and_then(|n| n.checked_add(GROUP_FIXED_BYTES + GROUP_LANE_BYTES))
                .ok_or(CoordinatorError::Overflow)?;
            if minimum > self.group_decoded_bytes as u64 {
                return Err(CoordinatorError::MinimumGroupBytes {
                    required: minimum,
                    limit: self.group_decoded_bytes,
                });
            }
        }
        Ok(row_cap)
    }
}

/// Buffers for the column types present in the captured schema. Unused type
/// slices may be empty. Capacity and ledger ownership remain with the caller.
pub struct ColumnScratch<'a> {
    pub spans: &'a mut [CellSpan],
    pub nulls: &'a mut [bool],
    pub integers: &'a mut [i64],
    pub floats: &'a mut [f64],
    pub booleans: &'a mut [bool],
    pub timestamps: &'a mut [DateTime<Utc>],
    pub bytes: &'a mut [u8],
    pub offsets: &'a mut [(u64, u64)],
}
impl ColumnScratch<'_> {
    fn validate(&self, columns: &[ColumnSpec], rows: usize, byte_cap: usize) -> Result<()> {
        if self.spans.len() < rows || self.nulls.len() < rows {
            return Err(CoordinatorError::BufferTooSmall);
        }
        if rows == 0 {
            return Ok(());
        }
        for column in columns {
            let fits = match column.data_type {
                DataType::Null => true,
                DataType::Integer => self.integers.len() >= rows,
                DataType::Float => self.floats.len() >= rows,
                DataType::Boolean => self.booleans.len() >= rows,
                DataType::Timestamp => self.timestamps.len() >= rows,
                DataType::Text | DataType::Json | DataType::Vector => {
                    self.bytes.len() >= byte_cap && self.offsets.len() >= rows
                }
            };
            if !fits {
                return Err(CoordinatorError::BufferTooSmall);
            }
        }
        Ok(())
    }
    fn gather<'out, R: ReadAt + ?Sized>(
        &'out mut self,
        group: &mut PreparedGroup<'_, '_, '_, R>,
        column: u32,
        data_type: DataType,
    ) -> Result<GatheredColumn<'out>> {
        let buffer = match data_type {
            DataType::Null => ColumnBuffer::AllNull,
            DataType::Integer => ColumnBuffer::I64(self.integers),
            DataType::Float => ColumnBuffer::F64(self.floats),
            DataType::Boolean => ColumnBuffer::Bool(self.booleans),
            DataType::Timestamp => ColumnBuffer::Timestamps(self.timestamps),
            DataType::Text | DataType::Json | DataType::Vector => ColumnBuffer::Variable {
                bytes: self.bytes,
                offsets: self.offsets,
            },
        };
        Ok(group.gather_column(column, self.spans, self.nulls, buffer)?)
    }
}

pub struct PlanningScratch<'a> {
    pub positions: &'a mut [RowPosition],
    /// One conservative payload/lane byte bound per row, never all-column metadata.
    pub prefix_costs: &'a mut [u64],
    pub sorted_io: &'a mut [u8],
    pub payload_window: &'a mut [u8],
    pub plan_output: &'a mut [u8],
    pub column: ColumnScratch<'a>,
}

#[derive(Clone, Copy)]
pub struct FinishedPlan<'schema> {
    runs: RowRuns<'schema>,
    config: BuildConfig,
    shape: VolumeShape,
    checksum: u32,
}
impl<'schema> FinishedPlan<'schema> {
    pub const fn shape(self) -> VolumeShape {
        self.shape
    }
    pub const fn config(self) -> BuildConfig {
        self.config
    }
    pub const fn spool(self) -> FinishedSpool<'schema> {
        self.runs.spool()
    }
    pub const fn byte_len(self) -> u64 {
        self.shape.group_count * PLAN_RECORD_BYTES as u64
    }
}

/// Full planning pass. Inputs and output must have distinct immutable/exclusive
/// backing. This rewinds only the caller's plan stream and ignores its old tail.
pub fn plan_groups<'schema, R: ReadAt + ?Sized, P: ReadAt + ?Sized, W: Write + Seek + ?Sized>(
    sorted_source: &R,
    payload_source: &P,
    plan_sink: &mut W,
    runs: RowRuns<'schema>,
    config: BuildConfig,
    scratch: &mut PlanningScratch<'_>,
) -> Result<FinishedPlan<'schema>> {
    let spool = runs.spool();
    let capacity = config.validate(spool)?;
    check_io(scratch.sorted_io)?;
    check_io(scratch.payload_window)?;
    check_io(scratch.plan_output)?;
    if scratch.positions.len() < capacity || scratch.prefix_costs.len() < capacity {
        return Err(CoordinatorError::BufferTooSmall);
    }
    let byte_cap = config.group_decoded_bytes.min(config.columns.decoded_bytes);
    scratch
        .column
        .validate(spool.binding().columns, capacity, byte_cap)?;
    let mut sorted = SortedRows::new(sorted_source, runs, scratch.sorted_io)?;
    let mut reader = SpoolReader::new(payload_source, spool, scratch.payload_window)?;
    if plan_sink
        .seek(SeekFrom::Start(0))
        .map_err(SpoolError::from)?
        != 0
    {
        return Err(CoordinatorError::Identity);
    }
    let mut output = PlanOutput {
        sink: plan_sink,
        buffer: scratch.plan_output,
        len: 0,
        checksum: crc32fast::Hasher::new(),
    };
    let mut seen = 0u64;
    let mut groups = 0u64;
    let mut bounds: Option<RowBounds> = None;
    while seen < spool.row_count() {
        let filled = (spool.row_count() - seen).min(capacity as u64) as usize;
        for slot in &mut scratch.positions[..filled] {
            *slot = sorted.next_position()?.ok_or(CoordinatorError::Coverage)?;
        }
        let mut batch = reader.prepare_group(&scratch.positions[..filled])?;
        let overhead = bound_rows(
            &mut batch,
            &scratch.positions[..filled],
            spool.binding().columns,
            config,
            &mut scratch.prefix_costs[..filled],
            scratch.column.spans,
        )?;
        let mut start = 0;
        while start < filled {
            let positions = &scratch.positions[start..filled];
            let limit = config.group_decoded_bytes.min(config.columns.decoded_bytes) as u64;
            let mut lanes = 0u64;
            let mut count = 0usize;
            for &bound in &scratch.prefix_costs[start..filled] {
                lanes = lanes.checked_add(bound).ok_or(CoordinatorError::Overflow)?;
                let nulls = (count as u64 + 1).div_ceil(8) * spool.binding().columns.len() as u64;
                if overhead
                    .checked_add(lanes)
                    .and_then(|n| n.checked_add(nulls))
                    .ok_or(CoordinatorError::Overflow)?
                    > limit
                {
                    break;
                }
                count += 1;
            }
            // Timestamp12 bounds may exceed the real narrow encoding. Always
            // try an exact single row before reporting that it cannot fit.
            count = count.max(1);
            let mut group = batch.range(start, count)?;
            let decoded_bytes = exact_group_bytes(
                &mut group,
                &positions[..count],
                spool.binding().columns,
                config,
                groups,
                seen,
                &mut scratch.column,
            )?;
            let rows = RowBounds {
                min: positions[0].row_id(),
                max: positions[count - 1].row_id(),
            };
            let record = PlannedGroup {
                ordinal: groups,
                record: GroupRecord {
                    row_start: seen,
                    row_count: count as u32,
                    column_count: spool.binding().columns.len() as u32,
                    rows,
                },
                decoded_bytes,
            };
            output.push(record)?;
            bounds = Some(RowBounds {
                min: bounds.map_or(rows.min, |b| b.min),
                max: rows.max,
            });
            seen += count as u64;
            start += count;
            groups += 1;
        }
    }
    if sorted.next_position()?.is_some() {
        return Err(CoordinatorError::Coverage);
    }
    let checksum = output.finish()?;
    Ok(FinishedPlan {
        runs,
        config,
        shape: VolumeShape {
            layout: Layout::RowId,
            row_count: spool.row_count(),
            column_count: spool.binding().columns.len() as u32,
            group_count: groups,
            rows: bounds,
            window: None,
        },
        checksum,
    })
}

// Fixed column headers, terminal variable offsets and rounded NULL maps are
// group overhead. Only timestamp lanes use a conservative width (12 bytes), so
// ordinary fixed/variable groups retain exact packing without all-column state.
fn bound_rows<R: ReadAt + ?Sized>(
    group: &mut PreparedGroup<'_, '_, '_, R>,
    rows: &[RowPosition],
    specs: &[ColumnSpec],
    config: BuildConfig,
    costs: &mut [u64],
    spans: &mut [CellSpan],
) -> Result<u64> {
    let mut lanes = GROUP_LANE_BYTES;
    let mut overhead = GROUP_FIXED_BYTES + 64 * specs.len() as u64;
    for spec in specs {
        lanes += match spec.data_type {
            DataType::Null => 0,
            DataType::Integer | DataType::Float => 8,
            DataType::Timestamp => 12,
            DataType::Boolean => 1,
            DataType::Text | DataType::Json | DataType::Vector => {
                overhead += 4;
                4
            }
        };
    }
    costs.fill(lanes);
    for (column, spec) in specs.iter().enumerate() {
        if matches!(
            spec.data_type,
            DataType::Text | DataType::Json | DataType::Vector
        ) {
            let spans = group.column_spans(column as u32, spans)?;
            for ((cost, span), row) in costs.iter_mut().zip(spans).zip(rows) {
                if span.byte_len() > config.columns.cell_bytes as u64 {
                    return Err(CoordinatorError::CannotFitRow {
                        row_id: row.row_id(),
                        decoded_limit: config.columns.cell_bytes,
                    });
                }
                *cost = cost
                    .checked_add(span.byte_len())
                    .ok_or(CoordinatorError::Overflow)?;
            }
        }
    }
    Ok(overhead)
}
fn exact_group_bytes<R: ReadAt + ?Sized>(
    group: &mut PreparedGroup<'_, '_, '_, R>,
    positions: &[RowPosition],
    specs: &[ColumnSpec],
    config: BuildConfig,
    ordinal: u64,
    row_start: u64,
    scratch: &mut ColumnScratch<'_>,
) -> Result<u64> {
    let mut decoded = GROUP_FIXED_BYTES + GROUP_LANE_BYTES * positions.len() as u64;
    for (column, spec) in specs.iter().enumerate() {
        // The bound already rejected multirow candidates whose payload would
        // exceed scratch. The exact-single-row fallback still needs this guard.
        if matches!(
            spec.data_type,
            DataType::Text | DataType::Json | DataType::Vector
        ) {
            let spans = group.column_spans(column as u32, scratch.spans)?;
            let bytes = spans
                .iter()
                .try_fold(0u64, |n, span| n.checked_add(span.byte_len()))
                .ok_or(CoordinatorError::Overflow)?;
            if bytes > scratch.bytes.len() as u64 || bytes > config.columns.decoded_bytes as u64 {
                return Err(CoordinatorError::CannotFitRow {
                    row_id: positions[0].row_id(),
                    decoded_limit: config.group_decoded_bytes,
                });
            }
        }
        let gathered = scratch.gather(group, column as u32, spec.data_type)?;
        let expected = ColumnExpectation {
            identity: ColumnIdentity {
                physical_column: column as u32,
                group: ordinal,
                row_start,
                row_count: positions.len() as u32,
                data_type: spec.data_type,
            },
            vector_dimensions: spec.vector_dimensions,
        };
        let plan = ColumnEncodePlan::new(expected, gathered.nulls, gathered.input, config.columns)
            .map_err(|error| match error {
                ColumnError::ByteLimit | ColumnError::CellLimit => CoordinatorError::CannotFitRow {
                    row_id: positions[0].row_id(),
                    decoded_limit: config.group_decoded_bytes,
                },
                error => CoordinatorError::Column(error),
            })?;
        decoded = decoded
            .checked_add(plan.encoded_len() as u64)
            .ok_or(CoordinatorError::Overflow)?;
    }
    if decoded > config.group_decoded_bytes as u64 {
        return Err(CoordinatorError::CannotFitRow {
            row_id: positions[0].row_id(),
            decoded_limit: config.group_decoded_bytes,
        });
    }
    Ok(decoded)
}

pub struct EmissionScratch<'a> {
    pub positions: &'a mut [RowPosition],
    pub row_ids: &'a mut [i64],
    pub sources: &'a mut [RowSource],
    pub sorted_io: &'a mut [u8],
    pub payload_window: &'a mut [u8],
    pub plan_input: &'a mut [u8],
    pub encoding: &'a mut [u8],
    pub column: ColumnScratch<'a>,
}

/// Emit planned Raw payload pages. The caller's destination must be empty at
/// offset zero, and the descriptor sink exclusive and empty. Later directory
/// completion/publication remains the existing separate checked operation.
#[allow(clippy::too_many_arguments)]
pub fn emit_payloads<
    'sink,
    R: ReadAt + ?Sized,
    P: ReadAt + ?Sized,
    G: ReadAt + ?Sized,
    W: Write + ?Sized,
    D: DescriptorSink + ?Sized,
>(
    sorted_source: &R,
    payload_source: &P,
    group_source: &G,
    sink: &'sink mut W,
    header: Header,
    checkpoint: Option<&PlannedCheckpoint>,
    plan: FinishedPlan<'_>,
    descriptors: &mut D,
    scratch: &mut EmissionScratch<'_>,
) -> Result<CompletedPayloads<'sink, W>> {
    let spool = plan.runs.spool();
    let capacity = plan.config.validate(spool)?;
    if header.identity != spool.binding().identity
        || SourceEncodingContext::for_staged_file(&header, checkpoint)
            .map_err(|_| CoordinatorError::Identity)?
            .legacy_base()
            != spool.binding().source.legacy_base()
    {
        return Err(CoordinatorError::Identity);
    }
    check_io(scratch.sorted_io)?;
    check_io(scratch.payload_window)?;
    check_io(scratch.plan_input)?;
    if scratch.positions.len() < capacity
        || scratch.row_ids.len() < capacity
        || scratch.sources.len() < capacity
        || scratch.encoding.len()
            < plan
                .config
                .columns
                .decoded_bytes
                .max(64)
                .max(32 + capacity * 8)
    {
        return Err(CoordinatorError::BufferTooSmall);
    }
    scratch.column.validate(
        spool.binding().columns,
        capacity,
        plan.config
            .group_decoded_bytes
            .min(plan.config.columns.decoded_bytes),
    )?;
    let mut sorted = SortedRows::new(sorted_source, plan.runs, scratch.sorted_io)?;
    let mut reader = SpoolReader::new(payload_source, spool, scratch.payload_window)?;
    let mut groups = PlanReader {
        source: group_source,
        plan,
        buffer: scratch.plan_input,
        at: 0,
        len: 0,
        ordinal: 0,
        seen: 0,
        previous: None,
        row_limit: capacity,
        checksum: crc32fast::Hasher::new(),
    };
    let mut writer = PayloadWriter::new(
        sink,
        header,
        plan.shape,
        checkpoint,
        spool.binding().columns,
        plan.config.pages,
        plan.config.columns,
        descriptors,
    )?;
    while let Some(group) = groups.next()? {
        let count = group.record.row_count as usize;
        for index in 0..count {
            let row = sorted.next_position()?.ok_or(CoordinatorError::Coverage)?;
            scratch.positions[index] = row;
            scratch.row_ids[index] = row.row_id();
            scratch.sources[index] = row.source();
        }
        if scratch.row_ids[0] != group.record.rows.min
            || scratch.row_ids[count - 1] != group.record.rows.max
        {
            return Err(CoordinatorError::Coverage);
        }
        let mut prepared = reader.prepare_group(&scratch.positions[..count])?;
        writer.begin_group(
            group.record,
            &scratch.row_ids[..count],
            &scratch.sources[..count],
            scratch.encoding,
        )?;
        let mut decoded = GROUP_FIXED_BYTES + GROUP_LANE_BYTES * count as u64;
        for (column, spec) in spool.binding().columns.iter().enumerate() {
            let gathered = scratch
                .column
                .gather(&mut prepared, column as u32, spec.data_type)?;
            let expected = ColumnExpectation {
                identity: ColumnIdentity {
                    physical_column: column as u32,
                    group: group.ordinal,
                    row_start: group.record.row_start,
                    row_count: count as u32,
                    data_type: spec.data_type,
                },
                vector_dimensions: spec.vector_dimensions,
            };
            decoded = decoded
                .checked_add(
                    ColumnEncodePlan::new(
                        expected,
                        gathered.nulls,
                        gathered.input,
                        plan.config.columns,
                    )?
                    .encoded_len() as u64,
                )
                .ok_or(CoordinatorError::Overflow)?;
            writer.write_column(gathered.nulls, gathered.input, scratch.encoding)?;
        }
        if decoded != group.decoded_bytes {
            return Err(CoordinatorError::Coverage);
        }
    }
    if sorted.next_position()?.is_some() {
        return Err(CoordinatorError::Coverage);
    }
    Ok(writer.finish_payloads()?)
}

#[derive(Clone, Copy)]
struct PlannedGroup {
    ordinal: u64,
    record: GroupRecord,
    decoded_bytes: u64,
}
impl PlannedGroup {
    fn encode(self) -> [u8; PLAN_RECORD_BYTES] {
        let mut b = [0; PLAN_RECORD_BYTES];
        b[..8].copy_from_slice(b"V5GP\x01\x00\x40\x00");
        b[8..16].copy_from_slice(&self.ordinal.to_le_bytes());
        b[16..24].copy_from_slice(&self.record.row_start.to_le_bytes());
        b[24..28].copy_from_slice(&self.record.row_count.to_le_bytes());
        b[28..32].copy_from_slice(&self.record.column_count.to_le_bytes());
        b[32..40].copy_from_slice(&self.record.rows.min.to_le_bytes());
        b[40..48].copy_from_slice(&self.record.rows.max.to_le_bytes());
        b[48..56].copy_from_slice(&self.decoded_bytes.to_le_bytes());
        let crc = crc32fast::hash(&b[..60]);
        b[60..].copy_from_slice(&crc.to_le_bytes());
        b
    }
    fn decode(b: &[u8]) -> Result<Self> {
        if crc32fast::hash(&b[..60]) != u32_at(b, 60) {
            return Err(CoordinatorError::PlanChecksum);
        }
        if &b[..8] != b"V5GP\x01\x00\x40\x00" || b[56..60] != [0; 4] {
            return Err(CoordinatorError::PlanRecord);
        }
        Ok(Self {
            ordinal: u64_at(b, 8),
            record: GroupRecord {
                row_start: u64_at(b, 16),
                row_count: u32_at(b, 24),
                column_count: u32_at(b, 28),
                rows: RowBounds {
                    min: i64_at(b, 32),
                    max: i64_at(b, 40),
                },
            },
            decoded_bytes: u64_at(b, 48),
        })
    }
}
struct PlanOutput<'a, W: Write + ?Sized> {
    sink: &'a mut W,
    buffer: &'a mut [u8],
    len: usize,
    checksum: crc32fast::Hasher,
}
impl<W: Write + ?Sized> PlanOutput<'_, W> {
    fn push(&mut self, group: PlannedGroup) -> Result<()> {
        if self.buffer.len() - self.len < PLAN_RECORD_BYTES {
            self.flush()?;
        }
        let bytes = group.encode();
        self.checksum.update(&bytes);
        self.buffer[self.len..self.len + PLAN_RECORD_BYTES].copy_from_slice(&bytes);
        self.len += PLAN_RECORD_BYTES;
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        write_all(self.sink, &self.buffer[..self.len])?;
        self.len = 0;
        Ok(())
    }
    fn finish(mut self) -> Result<u32> {
        self.flush()?;
        Ok(self.checksum.finalize())
    }
}
struct PlanReader<'a, 'schema, R: ReadAt + ?Sized> {
    source: &'a R,
    plan: FinishedPlan<'schema>,
    buffer: &'a mut [u8],
    at: usize,
    len: usize,
    ordinal: u64,
    seen: u64,
    previous: Option<i64>,
    row_limit: usize,
    checksum: crc32fast::Hasher,
}
impl<R: ReadAt + ?Sized> PlanReader<'_, '_, R> {
    fn next(&mut self) -> Result<Option<PlannedGroup>> {
        if self.ordinal == self.plan.shape.group_count {
            if self.seen != self.plan.shape.row_count
                || self.checksum.clone().finalize() != self.plan.checksum
            {
                return Err(CoordinatorError::Coverage);
            }
            return Ok(None);
        }
        if self.at == self.len {
            let records = (self.plan.shape.group_count - self.ordinal)
                .min((self.buffer.len() / PLAN_RECORD_BYTES) as u64)
                as usize;
            self.len = 0;
            read_exact(
                self.source,
                self.ordinal * PLAN_RECORD_BYTES as u64,
                &mut self.buffer[..records * PLAN_RECORD_BYTES],
            )?;
            self.at = 0;
            self.len = records * PLAN_RECORD_BYTES;
        }
        let bytes = &self.buffer[self.at..self.at + PLAN_RECORD_BYTES];
        let group = PlannedGroup::decode(bytes)?;
        if group.ordinal != self.ordinal
            || group.record.row_start != self.seen
            || group.record.row_count == 0
            || group.record.row_count as usize > self.row_limit
            || group.record.column_count != self.plan.shape.column_count
            || group.record.rows.min > group.record.rows.max
            || self
                .previous
                .is_some_and(|max| max >= group.record.rows.min)
            || group.decoded_bytes > self.plan.config.group_decoded_bytes as u64
            || group.decoded_bytes
                < GROUP_FIXED_BYTES + GROUP_LANE_BYTES * u64::from(group.record.row_count)
            || self
                .seen
                .checked_add(u64::from(group.record.row_count))
                .is_none_or(|n| n > self.plan.shape.row_count)
        {
            return Err(CoordinatorError::PlanRecord);
        }
        self.checksum.update(bytes);
        self.at += PLAN_RECORD_BYTES;
        self.ordinal += 1;
        self.seen += u64::from(group.record.row_count);
        self.previous = Some(group.record.rows.max);
        Ok(Some(group))
    }
}
fn check_io(bytes: &[u8]) -> Result<()> {
    if !(64..=MAX_IO_BYTES).contains(&bytes.len()) {
        return Err(CoordinatorError::BufferTooSmall);
    }
    Ok(())
}
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}
fn i64_at(b: &[u8], at: usize) -> i64 {
    i64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests;
