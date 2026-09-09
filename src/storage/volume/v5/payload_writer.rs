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

//! Inactive, bounded RowId-layout payload producer. The caller owns immutable
//! projected schema/typed inputs, all encoding buffers, and descriptor spools.
//! This does not capture arena rows, install a checkpoint, or acknowledge WAL.
//! Column blocks retain the foundation's 4096-row/1MiB limits: oversized already
//! committed values and K5 capture/removal are activation gates, not fallbacks.
//!
//! Each group emits one metadata page, identity/source pages, then each column
//! exactly once. Metadata entries go directly to a caller-owned bounded sorter.
//! The completed producer checks the sorted key sequence and backward page
//! ranges before forming its bound root. The spool must be the exclusive output
//! of this build: descriptors do not contain a file identity or authenticate the
//! referenced payload. Swapping in another same-shaped spool is outside this
//! trusted caller boundary; payload readers still validate checksums/identities.

use std::fmt;
use std::io::Write;

use lz4_flex::block::CompressTable;

use crate::core::DataType;

use super::column_block::{
    ColumnEncodePlan, ColumnError, ColumnInput, ColumnLimits, HEADER_BYTES, MAX_BLOCK_ROWS,
    MAX_DECODED_BYTES,
};
use super::compression::{CompressionError, CompressionPlan};
use super::directory::{
    DirectoryError, DirectoryKey, GLOBAL_COLUMN, KEY_REQUIRED, Layout, LeafEntry, Section,
    VolumeShape,
};
use super::directory_writer::{
    DirectoryScratch, DirectoryWriteError, DirectoryWriter, ENCODING_BYTES, MAX_ENTRIES,
};
use super::envelope::{Codec, Header, LegacyBase, ReadLimits};
use super::group_metadata::{
    GROUP_HEADER_BYTES, GROUP_RECORD_BYTES, GroupError, GroupExpectation, GroupPageExpectation,
    GroupRangeValidator, GroupRecord,
};
use super::metadata_runs::{RunError, RunWriter};
use super::page_io::{FinishedVolume, PageIoError, PageWriter};
use super::row_identity::{
    IdentityError, IdentityPageExpectation, PlannedCheckpoint, RowSource, SourceEncodingContext,
};

/// Borrowed once from the caller's pinned schema projection. No Schema/Value
/// vector is built or cloned by the producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnSpec {
    pub data_type: DataType,
    pub vector_dimensions: Option<u16>,
}

/// Accept the exact descriptor once or fail. The normal implementation is the
/// bounded RunWriter; it must not be shared with another build. An error/unwind
/// after a page write makes the complete payload producer unpublishable.
pub trait DescriptorSink {
    fn push(&mut self, entry: LeafEntry) -> std::result::Result<(), RunError>;
}
impl<W: Write + ?Sized> DescriptorSink for RunWriter<'_, W> {
    fn push(&mut self, entry: LeafEntry) -> std::result::Result<(), RunError> {
        RunWriter::push(self, entry)
    }
}

#[derive(Debug)]
pub enum PayloadError {
    Directory(DirectoryError),
    DirectoryWrite(DirectoryWriteError),
    PageIo(PageIoError),
    Group(GroupError),
    Identity(IdentityError),
    Column(ColumnError),
    Compression(CompressionError),
    Run(RunError),
    Schema,
    Layout,
    Capacity,
    Sequence,
    Incomplete,
    BufferTooSmall,
    DescriptorMismatch,
    Poisoned,
    Finished,
}
impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V5 payload writer: {self:?}")
    }
}
impl std::error::Error for PayloadError {}
macro_rules! from_error {
    ($source:ty, $variant:ident) => {
        impl From<$source> for PayloadError {
            fn from(error: $source) -> Self {
                Self::$variant(error)
            }
        }
    };
}
from_error!(DirectoryError, Directory);
from_error!(DirectoryWriteError, DirectoryWrite);
from_error!(PageIoError, PageIo);
from_error!(GroupError, Group);
from_error!(IdentityError, Identity);
from_error!(ColumnError, Column);
from_error!(CompressionError, Compression);
from_error!(RunError, Run);
type Result<T> = std::result::Result<T, PayloadError>;

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
            Self::Poisoned => Err(PayloadError::Poisoned),
            Self::Finished => Err(PayloadError::Finished),
        }
    }
}

pub struct PayloadWriter<
    'sink,
    'schema,
    'descriptors,
    W: Write + ?Sized,
    D: DescriptorSink + ?Sized,
> {
    writer: PageWriter<'sink, W>,
    columns: &'schema [ColumnSpec],
    descriptors: &'descriptors mut D,
    shape: VolumeShape,
    source: SourceEncodingContext,
    limits: ColumnLimits,
    coverage: GroupRangeValidator,
    active: Option<GroupExpectation>,
    next_column: u32,
    next_group: u64,
    expected_entries: u64,
    state: State,
}
impl<'sink, 'schema, 'descriptors, W: Write + ?Sized, D: DescriptorSink + ?Sized>
    PayloadWriter<'sink, 'schema, 'descriptors, W, D>
{
    /// The file sink is empty at offset zero; the descriptor sink is exclusive
    /// and empty. All scalar/schema checks precede the first header write.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sink: &'sink mut W,
        header: Header,
        shape: VolumeShape,
        checkpoint: Option<&PlannedCheckpoint>,
        columns: &'schema [ColumnSpec],
        page_limits: ReadLimits,
        limits: ColumnLimits,
        descriptors: &'descriptors mut D,
    ) -> Result<Self> {
        shape.validate()?;
        if shape.layout != Layout::RowId {
            return Err(PayloadError::Layout);
        }
        if columns.len() as u64 != u64::from(shape.column_count)
            || columns.iter().any(|c| {
                c.data_type != DataType::Vector
                    && c.vector_dimensions
                        .is_some_and(|dimensions| dimensions != 0)
            })
        {
            return Err(PayloadError::Schema);
        }
        if limits.rows == 0 || limits.rows > MAX_BLOCK_ROWS {
            return Err(ColumnError::RowLimit.into());
        }
        if !(HEADER_BYTES..=MAX_DECODED_BYTES).contains(&limits.decoded_bytes) {
            return Err(ColumnError::ByteLimit.into());
        }
        if page_limits.page_stored_bytes < ENCODING_BYTES as u64
            || page_limits.page_decoded_bytes < ENCODING_BYTES as u64
        {
            return Err(DirectoryWriteError::PageLimitsTooSmall.into());
        }
        let expected_entries = shape
            .group_count
            .checked_mul(u64::from(shape.column_count) + 3)
            .filter(|&n| n <= MAX_ENTRIES)
            .ok_or(PayloadError::Capacity)?;
        let source = SourceEncodingContext::for_staged_file(&header, checkpoint)?;
        let coverage = GroupRangeValidator::for_shape(&shape)?;
        let writer = PageWriter::new(sink, header, page_limits)?;
        Ok(Self {
            writer,
            columns,
            descriptors,
            shape,
            source,
            limits,
            coverage,
            active: None,
            next_column: 0,
            next_group: 0,
            expected_entries,
            state: State::Open,
        })
    }

    pub fn position(&self) -> u64 {
        self.writer.position()
    }

    /// Validates both lane inputs using one reused buffer before any page or
    /// descriptor write. It repeats bounded encoding passes to avoid retaining
    /// two identity pages. Maximum required scratch is 32,800 bytes.
    pub fn begin_group(
        &mut self,
        record: GroupRecord,
        row_ids: &[i64],
        sources: &[RowSource],
        scratch: &mut [u8],
    ) -> Result<()> {
        self.state.check()?;
        if self.active.is_some() {
            return Err(PayloadError::Sequence);
        }
        let group = GroupExpectation::for_shape(&self.shape, self.next_group, record)?;
        if record.row_count > self.limits.rows {
            return Err(ColumnError::RowLimit.into());
        }
        let identity = IdentityPageExpectation::new(group);
        self.check_page_length(identity.decoded_len())?;
        identity.encode_row_ids(row_ids, scratch)?;
        identity.encode_sources(sources, self.source, scratch)?;
        let metadata = GroupPageExpectation::for_shape(&self.shape, self.next_group, 1)?;
        let mut encoded = [0; GROUP_HEADER_BYTES + GROUP_RECORD_BYTES];
        metadata.encode(std::slice::from_ref(&record), &mut encoded)?;
        // Coverage errors are sticky. Once accepted, any failure below leaves
        // this producer poisoned, including a caught panic from either sink.
        self.state = State::Poisoned;
        self.coverage.push(group)?;
        self.append(
            Section::GroupMetadata,
            GLOBAL_COLUMN,
            self.next_group,
            &encoded,
        )?;
        let len = identity.encode_row_ids(row_ids, scratch)?;
        self.append(
            Section::RowIds,
            GLOBAL_COLUMN,
            self.next_group,
            &scratch[..len],
        )?;
        let len = identity.encode_sources(sources, self.source, scratch)?;
        self.append(
            Section::SourceLsns,
            GLOBAL_COLUMN,
            self.next_group,
            &scratch[..len],
        )?;
        self.next_group += 1;
        self.next_column = 0;
        self.active = (record.column_count != 0).then_some(group);
        self.state = State::Open;
        Ok(())
    }

    /// The ordinal is implicit: exactly one block for each bound column, in
    /// ascending order. Invalid typed inputs/short buffers leave file and spool
    /// unchanged; output bytes are caller scratch, never privately allocated.
    pub fn write_column(
        &mut self,
        nulls: &[bool],
        input: ColumnInput<'_>,
        scratch: &mut [u8],
    ) -> Result<()> {
        self.write_column_encoded(nulls, input, scratch, None)
    }

    /// Optionally compress one complete typed block, reusing caller-owned LZ4
    /// state. Decoded input, maximum output capacity and table coexist and must
    /// all be reserved by the caller. Compression errors precede external IO;
    /// incompressible blocks use Raw when it fits the stored-page limit.
    pub fn write_column_compressed(
        &mut self,
        nulls: &[bool],
        input: ColumnInput<'_>,
        decoded: &mut [u8],
        compressed: &mut [u8],
        table: &mut CompressTable,
    ) -> Result<()> {
        self.write_column_encoded(nulls, input, decoded, Some((compressed, table)))
    }

    fn write_column_encoded(
        &mut self,
        nulls: &[bool],
        input: ColumnInput<'_>,
        decoded: &mut [u8],
        compression: Option<(&mut [u8], &mut CompressTable)>,
    ) -> Result<()> {
        self.state.check()?;
        let group = self.active.ok_or(PayloadError::Sequence)?;
        let spec = self.columns[self.next_column as usize];
        let expected = group.column(self.next_column, spec.data_type, spec.vector_dimensions)?;
        let plan = ColumnEncodePlan::new(expected, nulls, input, self.limits)?;
        let compression_plan = if compression.is_some() {
            Some(CompressionPlan::new(
                plan.encoded_len(),
                &self.writer.limits(),
            )?)
        } else {
            self.check_page_length(plan.encoded_len())?;
            None
        };
        let len = plan.encode_into(decoded)?;
        let (codec, bytes) = if let Some((output, table)) = compression {
            let block = compression_plan
                .unwrap()
                .compress(&decoded[..len], output, table)?;
            (block.codec(), block.bytes())
        } else {
            (Codec::Raw, &decoded[..len])
        };
        self.state = State::Poisoned;
        self.append_encoded(
            Section::ColumnBlocks,
            self.next_column,
            group.group(),
            codec,
            bytes,
            len,
        )?;
        self.next_column += 1;
        if self.next_column == self.shape.column_count {
            self.active = None;
        }
        self.state = State::Open;
        Ok(())
    }

    /// The descriptor sink borrow ends here. Finish/sort its exact runs before
    /// calling CompletedPayloads::finish; no successful footer exists yet.
    pub fn finish_payloads(mut self) -> Result<CompletedPayloads<'sink, W>> {
        self.state.check()?;
        if self.active.is_some() || self.next_group != self.shape.group_count {
            return Err(PayloadError::Incomplete);
        }
        self.coverage.finish()?;
        Ok(CompletedPayloads {
            writer: self.writer,
            shape: self.shape,
            base: self.source.legacy_base(),
            entries: self.expected_entries,
            state: State::Open,
        })
    }
    fn check_page_length(&self, length: usize) -> Result<()> {
        let limits = self.writer.limits();
        if length as u64 > limits.page_stored_bytes || length as u64 > limits.page_decoded_bytes {
            return Err(PayloadError::BufferTooSmall);
        }
        Ok(())
    }
    fn append(&mut self, section: Section, column: u32, group: u64, bytes: &[u8]) -> Result<()> {
        self.append_encoded(section, column, group, Codec::Raw, bytes, bytes.len())
    }
    fn append_encoded(
        &mut self,
        section: Section,
        column: u32,
        group: u64,
        codec: Codec,
        bytes: &[u8],
        decoded_len: usize,
    ) -> Result<()> {
        let page = self
            .writer
            .append_stored(codec, bytes, decoded_len as u64)?;
        self.descriptors.push(LeafEntry {
            key: key(section, column, group),
            page,
        })?;
        Ok(())
    }
}

/// Owns the original file writer and bound shape/context. No API exposes a
/// mutable raw writer or accepts a replacement RootSummary/file identity.
pub struct CompletedPayloads<'sink, W: Write + ?Sized> {
    writer: PageWriter<'sink, W>,
    shape: VolumeShape,
    base: Option<LegacyBase>,
    entries: u64,
    state: State,
}
impl<W: Write + ?Sized> CompletedPayloads<'_, W> {
    pub const fn descriptor_count(&self) -> u64 {
        self.entries
    }
    /// `next` normally calls SortedReader::next_entry on this build's exact
    /// completed runs. Missing/extra/wrong keys, invalid backward references and
    /// run IO errors fail closed. The callback may not substitute another file's
    /// same-shaped descriptors; the trusted spool boundary is documented above.
    pub fn finish(
        &mut self,
        mut next: impl FnMut() -> std::result::Result<Option<LeafEntry>, RunError>,
        scratch: &mut DirectoryScratch,
        encoding: &mut [u8],
    ) -> Result<FinishedVolume> {
        self.state.check()?;
        if encoding.len() < ENCODING_BYTES {
            return Err(PayloadError::BufferTooSmall);
        }
        self.state = State::Poisoned;
        let payload_end = self.writer.position();
        let mut directory = DirectoryWriter::new(&mut self.writer, scratch, encoding)?;
        for ordinal in 0..self.entries {
            let entry = next()?.ok_or(PayloadError::DescriptorMismatch)?;
            if entry.key != expected_key(self.shape, ordinal)
                || entry
                    .page
                    .offset
                    .checked_add(entry.page.stored_len)
                    .is_none_or(|end| end > payload_end)
            {
                return Err(PayloadError::DescriptorMismatch);
            }
            directory.push(entry)?;
        }
        if next()?.is_some() {
            return Err(PayloadError::DescriptorMismatch);
        }
        let built = directory.finish()?;
        let root = self
            .shape
            .into_root(built.entry_count, built.root, self.base)?;
        let result = self.writer.finish(&root)?;
        self.state = State::Finished;
        Ok(result)
    }
}

fn key(section: Section, column: u32, group: u64) -> DirectoryKey {
    DirectoryKey {
        section: section as u16,
        flags: KEY_REQUIRED,
        column,
        ordinal: group,
    }
}
fn expected_key(shape: VolumeShape, ordinal: u64) -> DirectoryKey {
    // Called only for ordinal < groups*(columns+3), and therefore groups>0.
    let columns = u64::from(shape.column_count) * shape.group_count;
    if ordinal < columns {
        return key(
            Section::ColumnBlocks,
            (ordinal / shape.group_count) as u32,
            ordinal % shape.group_count,
        );
    }
    let remaining = ordinal - columns;
    let section = match remaining / shape.group_count {
        0 => Section::RowIds,
        1 => Section::SourceLsns,
        _ => Section::GroupMetadata,
    };
    key(section, GLOBAL_COLUMN, remaining % shape.group_count)
}

#[cfg(test)]
mod tests;
