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

//! Explicit physical group ranges, with allocation-free local and streaming
//! validation. Revision 1 uses little-endian fields:
//!
//! Header32: V5GM[0..4], revision:u16[4..6], size:u16[6..8], first_group:u64[8..16],
//! count:u16[16..18], record_size:u16[18..20], actual_layout:u8[20], zero[21..32].
//! Each record32: row_start:u64[0..8], row_count:u32[8..12], column_count:u32[12..16],
//! min_row_id:i64[16..24], max_row_id:i64[24..32]. Group IDs are first_group+i.
//! Pages hold 1..64 records (at most 2080 decoded bytes); each group holds
//! 1..4096 rows. Directory key is (GroupMetadata, GLOBAL_COLUMN, first_group).
//! CRC/exact decompression precede parsing. Construct GroupPageExpectation and
//! its exact read plan before reserving buffers; no compressed header is trusted
//! to choose an allocation. Encoders preflight every record before output writes.
//!
//! Local validation proves bounded identities, ranges and extrema only.
//! GroupRangeValidator explicitly drains group IDs 0..root.group_count and
//! physical rows 0..root.row_count, with exact root extrema. RowId layout adds
//! strictly ordered disjoint row-ID ranges. Clustered ranges may overlap; global
//! row-ID uniqueness and locator bijection belong to the later identity/locator
//! validator, not this bounded metadata codec. No operation allocates.

use std::fmt;

use super::column_block::{ColumnExpectation, ColumnIdentity, MAX_BLOCK_ROWS};
use super::directory::{DirectoryError, Layout, RootSummary, RowBounds};
use super::envelope::{Footer, PageDescriptor, ReadLimits};
use super::page_io::{PageIoError, PageReadPlan};
use crate::core::DataType;

pub const GROUP_HEADER_BYTES: usize = 32;
pub const GROUP_RECORD_BYTES: usize = 32;
pub const MAX_GROUP_RECORDS: u16 = 64;
pub const MAX_GROUP_PAGE_BYTES: usize =
    GROUP_HEADER_BYTES + MAX_GROUP_RECORDS as usize * GROUP_RECORD_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupError {
    Directory(DirectoryError),
    Length,
    Magic,
    Revision,
    Reserved,
    Layout,
    Identity,
    Count,
    Range,
    Bounds,
    Columns,
    Overflow,
    OutputTooShort,
    Sequence,
    Incomplete,
    Finished,
}
impl fmt::Display for GroupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid V5 group metadata: {self:?}")
    }
}
impl std::error::Error for GroupError {}
impl From<DirectoryError> for GroupError {
    fn from(error: DirectoryError) -> Self {
        Self::Directory(error)
    }
}
type Result<T> = std::result::Result<T, GroupError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupRecord {
    pub row_start: u64,
    pub row_count: u32,
    pub column_count: u32,
    pub rows: RowBounds,
}

/// Locally validated metadata; it is not whole-volume coverage or a durability
/// proof. Private fields keep lane/column readers tied to this checked range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupExpectation {
    group: u64,
    layout: Layout,
    record: GroupRecord,
}
impl GroupExpectation {
    pub fn new(root: &RootSummary, group: u64, record: GroupRecord) -> Result<Self> {
        root.validate()?;
        validate_record(root, group, record)?;
        Ok(Self {
            group,
            layout: root.layout,
            record,
        })
    }
    pub const fn group(self) -> u64 {
        self.group
    }
    pub const fn layout(self) -> Layout {
        self.layout
    }
    pub const fn record(self) -> GroupRecord {
        self.record
    }
    pub fn column(
        self,
        physical_column: u32,
        data_type: DataType,
        vector_dimensions: Option<u16>,
    ) -> Result<ColumnExpectation> {
        if physical_column >= self.record.column_count || physical_column == u32::MAX {
            return Err(GroupError::Columns);
        }
        Ok(ColumnExpectation {
            identity: ColumnIdentity {
                physical_column,
                group: self.group,
                row_start: self.record.row_start,
                row_count: self.record.row_count,
                data_type,
            },
            vector_dimensions,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GroupPageExpectation {
    root: RootSummary,
    first_group: u64,
    count: u16,
}
impl GroupPageExpectation {
    pub fn new(root: &RootSummary, first_group: u64, count: u16) -> Result<Self> {
        root.validate()?;
        if !(1..=MAX_GROUP_RECORDS).contains(&count) {
            return Err(GroupError::Count);
        }
        if first_group
            .checked_add(u64::from(count))
            .ok_or(GroupError::Overflow)?
            > root.group_count
        {
            return Err(GroupError::Identity);
        }
        Ok(Self {
            root: *root,
            first_group,
            count,
        })
    }
    /// The descriptor, not an unread compressed header, determines bounded
    /// record count. Identity still comes from the expected directory key.
    pub fn for_descriptor(
        root: &RootSummary,
        first_group: u64,
        descriptor: PageDescriptor,
    ) -> Result<Self> {
        let n = descriptor
            .decoded_len
            .checked_sub(GROUP_HEADER_BYTES as u64)
            .ok_or(GroupError::Length)?;
        if !n.is_multiple_of(GROUP_RECORD_BYTES as u64) {
            return Err(GroupError::Length);
        }
        let count = u16::try_from(n / GROUP_RECORD_BYTES as u64).map_err(|_| GroupError::Count)?;
        Self::new(root, first_group, count)
    }
    pub const fn decoded_len(self) -> usize {
        GROUP_HEADER_BYTES + self.count as usize * GROUP_RECORD_BYTES
    }
    pub const fn count(self) -> u16 {
        self.count
    }
    pub const fn first_group(self) -> u64 {
        self.first_group
    }
    pub fn read_plan(
        self,
        footer: &Footer,
        descriptor: PageDescriptor,
        limits: &ReadLimits,
    ) -> std::result::Result<PageReadPlan, PageIoError> {
        if descriptor.decoded_len != self.decoded_len() as u64 {
            return Err(PageIoError::PayloadLength);
        }
        PageReadPlan::for_page(footer, descriptor, limits)
    }
    pub fn decode(self, bytes: &[u8]) -> Result<GroupMetadataRef<'_>> {
        if bytes.len() != self.decoded_len() {
            return Err(GroupError::Length);
        }
        if &bytes[..4] != b"V5GM" {
            return Err(GroupError::Magic);
        }
        if u16_at(bytes, 4) != 1 {
            return Err(GroupError::Revision);
        }
        if u16_at(bytes, 6) != GROUP_HEADER_BYTES as u16
            || u16_at(bytes, 18) != GROUP_RECORD_BYTES as u16
        {
            return Err(GroupError::Length);
        }
        if u64_at(bytes, 8) != self.first_group || u16_at(bytes, 16) != self.count {
            return Err(GroupError::Identity);
        }
        if bytes[20] != layout_tag(self.root.layout) {
            return Err(GroupError::Layout);
        }
        if bytes[21..32].iter().any(|b| *b != 0) {
            return Err(GroupError::Reserved);
        }
        let view = GroupMetadataRef {
            bytes,
            expected: self,
        };
        for item in view.iter() {
            validate_record(&self.root, item.group, item.record)?;
        }
        Ok(view)
    }
    pub fn encode(self, records: &[GroupRecord], output: &mut [u8]) -> Result<usize> {
        if records.len() != usize::from(self.count) {
            return Err(GroupError::Count);
        }
        if output.len() < self.decoded_len() {
            return Err(GroupError::OutputTooShort);
        }
        for (i, record) in records.iter().enumerate() {
            validate_record(&self.root, self.first_group + i as u64, *record)?;
        }
        let bytes = &mut output[..self.decoded_len()];
        bytes[..GROUP_HEADER_BYTES].fill(0);
        bytes[..4].copy_from_slice(b"V5GM");
        bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
        bytes[6..8].copy_from_slice(&(GROUP_HEADER_BYTES as u16).to_le_bytes());
        bytes[8..16].copy_from_slice(&self.first_group.to_le_bytes());
        bytes[16..18].copy_from_slice(&self.count.to_le_bytes());
        bytes[18..20].copy_from_slice(&(GROUP_RECORD_BYTES as u16).to_le_bytes());
        bytes[20] = layout_tag(self.root.layout);
        for (record, dest) in records.iter().zip(
            bytes[GROUP_HEADER_BYTES..]
                .as_chunks_mut::<GROUP_RECORD_BYTES>()
                .0,
        ) {
            dest[..8].copy_from_slice(&record.row_start.to_le_bytes());
            dest[8..12].copy_from_slice(&record.row_count.to_le_bytes());
            dest[12..16].copy_from_slice(&record.column_count.to_le_bytes());
            dest[16..24].copy_from_slice(&record.rows.min.to_le_bytes());
            dest[24..32].copy_from_slice(&record.rows.max.to_le_bytes());
        }
        Ok(bytes.len())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GroupMetadataRef<'a> {
    bytes: &'a [u8],
    expected: GroupPageExpectation,
}
impl GroupMetadataRef<'_> {
    pub fn iter(&self) -> impl ExactSizeIterator<Item = GroupExpectation> + '_ {
        self.bytes[GROUP_HEADER_BYTES..]
            .as_chunks::<GROUP_RECORD_BYTES>()
            .0
            .iter()
            .enumerate()
            .map(|(i, b)| GroupExpectation {
                group: self.expected.first_group + i as u64,
                layout: self.expected.root.layout,
                record: GroupRecord {
                    row_start: u64_at(b, 0),
                    row_count: u32_at(b, 8),
                    column_count: u32_at(b, 12),
                    rows: RowBounds {
                        min: i64_at(b, 16),
                        max: i64_at(b, 24),
                    },
                },
            })
    }
}

/// Scalar, sticky-error validation of explicit complete group coverage. Call
/// finish after all pages; a yielded prefix never proves complete coverage.
pub struct GroupRangeValidator {
    root: RootSummary,
    next_group: u64,
    next_row: u64,
    rows: Option<RowBounds>,
    previous_max: Option<i64>,
    error: Option<GroupError>,
    finished: bool,
}
impl GroupRangeValidator {
    pub fn new(root: &RootSummary) -> Result<Self> {
        root.validate()?;
        Ok(Self {
            root: *root,
            next_group: 0,
            next_row: 0,
            rows: None,
            previous_max: None,
            error: None,
            finished: false,
        })
    }
    pub fn push(&mut self, group: GroupExpectation) -> Result<()> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if self.finished {
            return Err(GroupError::Finished);
        }
        let result = self.push_checked(group);
        if let Err(error) = result {
            self.error = Some(error);
        }
        result
    }
    fn push_checked(&mut self, group: GroupExpectation) -> Result<()> {
        validate_record(&self.root, group.group, group.record)?;
        let record = group.record;
        if group.layout != self.root.layout
            || group.group != self.next_group
            || record.row_start != self.next_row
        {
            return Err(GroupError::Sequence);
        }
        if self.root.layout == Layout::RowId
            && self.previous_max.is_some_and(|max| max >= record.rows.min)
        {
            return Err(GroupError::Bounds);
        }
        // Bounds preflight precedes mutation; validated record cannot overflow.
        self.next_group += 1;
        self.next_row = record.row_start + u64::from(record.row_count);
        self.rows = Some(match self.rows {
            None => record.rows,
            Some(old) => RowBounds {
                min: old.min.min(record.rows.min),
                max: old.max.max(record.rows.max),
            },
        });
        self.previous_max = Some(record.rows.max);
        Ok(())
    }
    pub fn finish(&mut self) -> Result<()> {
        if let Some(error) = self.error {
            return Err(error);
        }
        if self.next_group != self.root.group_count
            || self.next_row != self.root.row_count
            || self.rows != self.root.rows
        {
            self.error = Some(GroupError::Incomplete);
            return Err(GroupError::Incomplete);
        }
        self.finished = true;
        Ok(())
    }
}

fn validate_record(root: &RootSummary, group: u64, record: GroupRecord) -> Result<()> {
    if group >= root.group_count {
        return Err(GroupError::Identity);
    }
    if record.row_count == 0 || record.row_count > MAX_BLOCK_ROWS {
        return Err(GroupError::Count);
    }
    if record
        .row_start
        .checked_add(u64::from(record.row_count))
        .ok_or(GroupError::Overflow)?
        > root.row_count
    {
        return Err(GroupError::Range);
    }
    if record.column_count != root.column_count {
        return Err(GroupError::Columns);
    }
    let bounds = root.rows.ok_or(GroupError::Bounds)?;
    if record.rows.min > record.rows.max
        || record.rows.min < bounds.min
        || record.rows.max > bounds.max
        || u128::from(record.row_count)
            > (i128::from(record.rows.max) - i128::from(record.rows.min) + 1) as u128
    {
        return Err(GroupError::Bounds);
    }
    Ok(())
}
pub(crate) const fn layout_tag(layout: Layout) -> u8 {
    match layout {
        Layout::RowId => 0,
        Layout::Clustered => 1,
    }
}
fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
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
pub(crate) mod tests {
    use super::super::directory::DirectoryRoot;
    use super::super::envelope::Codec;
    use super::*;

    pub(crate) fn root(
        layout: Layout,
        row_count: u64,
        group_count: u64,
        rows: RowBounds,
    ) -> RootSummary {
        RootSummary {
            layout,
            row_count,
            column_count: 2,
            group_count,
            entry_count: 1,
            rows: Some(rows),
            window: None,
            legacy_base: None,
            directory: Some(DirectoryRoot {
                depth: 1,
                page: PageDescriptor {
                    offset: 64,
                    stored_len: 80,
                    decoded_len: 80,
                    stored_checksum: 0,
                    codec: Codec::Raw,
                },
            }),
        }
    }
    pub(crate) fn fixed_hex<const N: usize>(text: &str) -> [u8; N] {
        assert_eq!(text.len(), 2 * N);
        std::array::from_fn(|i| u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).unwrap())
    }
    fn record(start: u64, count: u32, min: i64, max: i64) -> GroupRecord {
        GroupRecord {
            row_start: start,
            row_count: count,
            column_count: 2,
            rows: RowBounds { min, max },
        }
    }
    #[test]
    fn group_wire_golden_extremes_and_exact_decode() {
        // Fixed independent Python struct/zlib vector, including signed endpoints.
        let golden = fixed_hex::<64>("5635474d01002000000000000000000001002000000000000000000000000000000000000000000003000000020000000000000000000080ffffffffffffff7f");
        assert_eq!(crc32fast::hash(&golden), 0x9206_0fea);
        let root = root(
            Layout::RowId,
            3,
            1,
            RowBounds {
                min: i64::MIN,
                max: i64::MAX,
            },
        );
        let expected = GroupPageExpectation::new(&root, 0, 1).unwrap();
        let row = record(0, 3, i64::MIN, i64::MAX);
        let mut output = [0x5a; 80];
        assert_eq!(expected.encode(&[row], &mut output).unwrap(), 64);
        assert_eq!(output[..64], golden);
        assert_eq!(output[64..], [0x5a; 16]);
        let group = expected.decode(&golden).unwrap().iter().next().unwrap();
        assert_eq!(group.record(), row);
        let column = group.column(1, DataType::Integer, None).unwrap();
        assert_eq!(
            (
                column.identity.group,
                column.identity.row_start,
                column.identity.row_count
            ),
            (0, 0, 3)
        );
        assert_eq!(
            group.column(2, DataType::Integer, None),
            Err(GroupError::Columns)
        );
        let mut validator = GroupRangeValidator::new(&root).unwrap();
        validator.push(group).unwrap();
        validator.finish().unwrap();
        assert_eq!(validator.push(group), Err(GroupError::Finished));
        for len in 0..64 {
            assert!(expected.decode(&golden[..len]).is_err());
        }
        assert!(expected.decode(&output).is_err());
    }
    #[test]
    fn group_metadata_rejects_corruption_and_preserves_output_on_failure() {
        let root = root(Layout::RowId, 3, 1, RowBounds { min: -1, max: 1 });
        let expected = GroupPageExpectation::new(&root, 0, 1).unwrap();
        let mut bytes = [0; 64];
        expected.encode(&[record(0, 3, -1, 1)], &mut bytes).unwrap();
        for (offset, value, error) in [
            (0, 0, GroupError::Magic),
            (4, 2, GroupError::Revision),
            (6, 31, GroupError::Length),
            (8, 1, GroupError::Identity),
            (16, 0, GroupError::Identity),
            (18, 31, GroupError::Length),
            (20, 1, GroupError::Layout),
            (31, 1, GroupError::Reserved),
            (32, 1, GroupError::Range),
            (40, 0, GroupError::Count),
            (44, 1, GroupError::Columns),
            (48, 0, GroupError::Bounds),
        ] {
            let mut corrupt = bytes;
            corrupt[offset] = value;
            assert_eq!(
                expected.decode(&corrupt).unwrap_err(),
                error,
                "offset {offset}"
            );
        }
        let mut output = [0x5a; 64];
        assert_eq!(
            expected.encode(&[record(1, 3, -1, 1)], &mut output),
            Err(GroupError::Range)
        );
        assert_eq!(output, [0x5a; 64]);
        assert_eq!(expected.encode(&[], &mut output), Err(GroupError::Count));
        assert_eq!(
            expected.encode(&[record(0, 3, -1, 1)], &mut output[..63]),
            Err(GroupError::OutputTooShort)
        );
    }
    #[test]
    fn group_sequence_streams_page_boundary_and_rejects_gaps_duplicates_and_wrong_totals() {
        let root = root(Layout::RowId, 65, 65, RowBounds { min: -32, max: 32 });
        let mut validator = GroupRangeValidator::new(&root).unwrap();
        let mut bytes = [0; MAX_GROUP_PAGE_BYTES];
        let rows =
            std::array::from_fn::<_, 64, _>(|i| record(i as u64, 1, i as i64 - 32, i as i64 - 32));
        let page = GroupPageExpectation::new(&root, 0, 64).unwrap();
        let len = page.encode(&rows, &mut bytes).unwrap();
        assert_eq!(len, MAX_GROUP_PAGE_BYTES);
        for item in page.decode(&bytes).unwrap().iter() {
            validator.push(item).unwrap();
        }
        let tail = GroupExpectation::new(&root, 64, record(64, 1, 32, 32)).unwrap();
        validator.push(tail).unwrap();
        validator.finish().unwrap();
        for (group, r) in [(1, record(0, 1, -32, -32)), (0, record(1, 1, -32, -32))] {
            let mut invalid = GroupRangeValidator::new(&root).unwrap();
            let item = GroupExpectation::new(&root, group, r).unwrap();
            assert_eq!(invalid.push(item), Err(GroupError::Sequence));
            assert_eq!(invalid.finish(), Err(GroupError::Sequence));
        }
        let first = GroupExpectation::new(&root, 0, rows[0]).unwrap();
        let mut dup = GroupRangeValidator::new(&root).unwrap();
        dup.push(first).unwrap();
        assert_eq!(dup.push(first), Err(GroupError::Sequence));
        let mut incomplete = GroupRangeValidator::new(&root).unwrap();
        incomplete.push(first).unwrap();
        assert_eq!(incomplete.finish(), Err(GroupError::Incomplete));
        assert_eq!(incomplete.push(tail), Err(GroupError::Incomplete));
        let wrong_extrema = self::root(Layout::Clustered, 2, 1, RowBounds { min: -3, max: 3 });
        let mut invalid = GroupRangeValidator::new(&wrong_extrema).unwrap();
        invalid
            .push(GroupExpectation::new(&wrong_extrema, 0, record(0, 2, -1, 1)).unwrap())
            .unwrap();
        assert_eq!(invalid.finish(), Err(GroupError::Incomplete));
    }
    #[test]
    fn group_clustered_overlap_is_permitted_but_rowid_overlap_is_not() {
        for layout in [Layout::RowId, Layout::Clustered] {
            let root = root(layout, 4, 2, RowBounds { min: -5, max: 5 });
            let mut validator = GroupRangeValidator::new(&root).unwrap();
            validator
                .push(GroupExpectation::new(&root, 0, record(0, 2, -5, 2)).unwrap())
                .unwrap();
            let second = GroupExpectation::new(&root, 1, record(2, 2, -2, 5)).unwrap();
            if layout == Layout::RowId {
                assert_eq!(validator.push(second), Err(GroupError::Bounds));
            } else {
                validator.push(second).unwrap();
                validator.finish().unwrap();
            }
        }
    }
    #[test]
    fn group_read_plan_checks_caps_and_arithmetic_before_reservation() {
        let mut root = root(
            Layout::RowId,
            u64::MAX,
            u64::MAX,
            RowBounds {
                min: i64::MIN,
                max: i64::MAX,
            },
        );
        assert_eq!(
            GroupPageExpectation::new(&root, u64::MAX, 1).unwrap_err(),
            GroupError::Overflow
        );
        assert_eq!(
            GroupExpectation::new(&root, 0, record(u64::MAX, 1, 0, 0)),
            Err(GroupError::Overflow)
        );
        assert_eq!(
            GroupExpectation::new(&root, 0, record(0, 4097, 0, 4096)),
            Err(GroupError::Count)
        );
        let page = root.directory.unwrap().page;
        for decoded_len in [0, 31, 32, 63, 65, 2081, 2112, u64::MAX] {
            assert!(GroupPageExpectation::for_descriptor(
                &root,
                0,
                PageDescriptor {
                    decoded_len,
                    ..page
                }
            )
            .is_err());
        }
        let expected = GroupPageExpectation::new(&root, 0, 1).unwrap();
        let limits = ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: 4096,
            page_decoded_bytes: 4096,
        };
        let footer = Footer {
            file_length: 8192 + 192,
            root: PageDescriptor {
                offset: 8192,
                stored_len: 128,
                decoded_len: 128,
                ..page
            },
        };
        assert!(matches!(
            expected.read_plan(&footer, page, &limits),
            Err(PageIoError::PayloadLength)
        ));
        let exact = PageDescriptor {
            stored_len: 64,
            decoded_len: 64,
            ..page
        };
        let plan = expected.read_plan(&footer, exact, &limits).unwrap();
        assert_eq!(
            (plan.stored_buffer_len(), plan.decoded_buffer_len()),
            (64, 0)
        );
        root.row_count = 0;
        root.group_count = 0;
        root.entry_count = 0;
        root.rows = None;
        root.directory = None;
        GroupRangeValidator::new(&root).unwrap().finish().unwrap();
        assert!(GroupPageExpectation::new(&root, 0, 1).is_err());
    }
}
