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

use super::*;
use std::io::{self, Cursor};
use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};

use super::super::column_block::{ColumnBlockRef, ColumnCell};
use super::super::directory::{RootSummary, RowBounds};
use super::super::directory_reader::DirectoryWalker;
use super::super::envelope::{FileIdentity, REQUIRED_LEGACY_BASE};
use super::super::metadata_runs::{merge_pass, MergeScratch, SortedReader, IO_BYTES};
use super::super::page_io::{OpenedEnvelope, PageReadPlan, ReadAt};
use super::super::row_identity::{SourceContext, VerifiedLegacyBase};

const SPEC: [ColumnSpec; 1] = [ColumnSpec {
    data_type: DataType::Integer,
    vector_dimensions: None,
}];
fn header() -> Header {
    Header::new(FileIdentity::new(1, 2, 3).unwrap())
}
fn limits() -> ReadLimits {
    ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: 1024 * 1024,
        page_decoded_bytes: 1024 * 1024,
    }
}
fn shape(groups: u64) -> VolumeShape {
    VolumeShape {
        layout: Layout::RowId,
        row_count: groups,
        column_count: 1,
        group_count: groups,
        rows: (groups != 0).then_some(RowBounds {
            min: 0,
            max: groups as i64 - 1,
        }),
        window: None,
    }
}
fn record(group: u64) -> GroupRecord {
    GroupRecord {
        row_start: group,
        row_count: 1,
        column_count: 1,
        rows: RowBounds {
            min: group as i64,
            max: group as i64,
        },
    }
}
fn source() -> RowSource {
    RowSource::Dml(NonZeroU64::new(7).unwrap())
}
fn empty_entry() -> LeafEntry {
    LeafEntry {
        key: key(Section::RowIds, GLOBAL_COLUMN, 0),
        page: super::super::envelope::PageDescriptor {
            offset: 64,
            stored_len: 1,
            decoded_len: 1,
            stored_checksum: 0,
            codec: Codec::Raw,
        },
    }
}

#[derive(Default)]
struct Entries(Vec<LeafEntry>);
impl DescriptorSink for Entries {
    fn push(&mut self, e: LeafEntry) -> std::result::Result<(), RunError> {
        self.0.push(e);
        Ok(())
    }
}
struct Bytes<'a>(&'a [u8]);
impl ReadAt for Bytes<'_> {
    fn read_at(&self, offset: u64, output: &mut [u8]) -> io::Result<usize> {
        let at = usize::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?;
        let bytes = self.0.get(at..).ok_or(io::ErrorKind::UnexpectedEof)?;
        let n = bytes.len().min(output.len());
        output[..n].copy_from_slice(&bytes[..n]);
        Ok(n)
    }
}

#[test]
fn exact_groups_spill_sort_finish_and_read_actual_payloads() {
    for groups in [0, 1, 65] {
        let mut file = Vec::new();
        let mut first = Cursor::new(Vec::new());
        let mut second = Cursor::new(Vec::new());
        let mut entries = [empty_entry(); 3];
        let mut run_encoding = [0; IO_BYTES];
        let mut runs = RunWriter::new(&mut first, &mut entries, &mut run_encoding).unwrap();
        let shape = shape(groups);
        let mut writer = PayloadWriter::new(
            &mut file,
            header(),
            shape,
            None,
            &SPEC,
            limits(),
            ColumnLimits::default(),
            &mut runs,
        )
        .unwrap();
        let mut encoding = [0; 128];
        for group in 0..groups {
            writer
                .begin_group(record(group), &[group as i64], &[source()], &mut encoding)
                .unwrap();
            writer
                .write_column(
                    &[false],
                    ColumnInput::I64(&[100 + group as i64]),
                    &mut encoding,
                )
                .unwrap();
        }
        let mut complete = writer.finish_payloads().unwrap();
        assert_eq!(complete.descriptor_count(), groups * 4);
        let mut descriptor_runs = runs.finish().unwrap();
        let mut merge = MergeScratch::new();
        while !descriptor_runs.is_sorted() {
            descriptor_runs = merge_pass(
                &Bytes(first.get_ref()),
                &mut second,
                descriptor_runs,
                &mut merge,
            )
            .unwrap();
            std::mem::swap(&mut first, &mut second);
        }
        let spool = Bytes(first.get_ref());
        let mut read_buffer = [0; IO_BYTES];
        let mut sorted = SortedReader::new(&spool, descriptor_runs, &mut read_buffer).unwrap();
        let mut directory = DirectoryScratch::new();
        let mut directory_encoding = [0; ENCODING_BYTES];
        let finished = complete
            .finish(
                || sorted.next_entry(),
                &mut directory,
                &mut directory_encoding,
            )
            .unwrap();
        assert_eq!(finished.summary.shape(), shape);
        assert!(matches!(
            complete.finish(|| Ok(None), &mut directory, &mut directory_encoding),
            Err(PayloadError::Finished)
        ));
        let disk = Bytes(&file);
        let opened = OpenedEnvelope::read(&disk, file.len() as u64, &limits()).unwrap();
        opened.require_identity(header().identity).unwrap();
        let root_plan = PageReadPlan::for_root(&opened.footer, &limits()).unwrap();
        let mut root_bytes = [0; 128];
        root_plan
            .read_into(&disk, &mut root_bytes, &mut [])
            .unwrap();
        let root = RootSummary::decode(&root_bytes, &opened.footer, &limits()).unwrap();
        assert_eq!(root, finished.summary);
        let context = SourceContext::bind(&opened.header, &root, None).unwrap();
        let mut sorted = SortedReader::new(&spool, descriptor_runs, &mut read_buffer).unwrap();
        let mut nodes = [[0; super::super::directory::MAX_NODE_BYTES]; 2];
        let mut stored_node = [0; super::super::directory::MAX_NODE_BYTES];
        let mut walker = DirectoryWalker::new(
            &disk,
            opened.footer,
            root,
            limits(),
            &mut nodes,
            &mut stored_node,
        )
        .unwrap();
        while let Some(entry) = sorted.next_entry().unwrap() {
            assert_eq!(walker.next_entry().unwrap(), Some(entry));
            let group =
                GroupExpectation::new(&root, entry.key.ordinal, record(entry.key.ordinal)).unwrap();
            let plan = PageReadPlan::for_page(&opened.footer, entry.page, &limits()).unwrap();
            let mut page = [0; 128];
            let page = plan.read_into(&disk, &mut page, &mut []).unwrap();
            match entry.key.section {
                n if n == Section::ColumnBlocks as u16 => {
                    let expected = group.column(0, DataType::Integer, None).unwrap();
                    let block =
                        ColumnBlockRef::parse(page, expected, ColumnLimits::default()).unwrap();
                    assert_eq!(
                        block.cell(0),
                        Some(ColumnCell::Integer(100 + entry.key.ordinal as i64))
                    );
                }
                n if n == Section::RowIds as u16 => assert_eq!(
                    IdentityPageExpectation::new(group)
                        .decode_row_ids(page)
                        .unwrap()
                        .iter()
                        .next(),
                    Some(entry.key.ordinal as i64)
                ),
                n if n == Section::SourceLsns as u16 => assert_eq!(
                    IdentityPageExpectation::new(group)
                        .decode_sources(page, context)
                        .unwrap()
                        .iter()
                        .next(),
                    Some(source())
                ),
                _ => {
                    GroupPageExpectation::new(&root, entry.key.ordinal, 1)
                        .unwrap()
                        .decode(page)
                        .unwrap();
                }
            }
        }
        assert_eq!(walker.next_entry().unwrap(), None);
        walker.finish_validation().unwrap();
    }
}

#[test]
fn preheader_shape_schema_context_and_capacity_fail_without_io() {
    let mut sink = Vec::new();
    let mut entries = Entries::default();
    let mut variants = [shape(1); 3];
    variants[0].layout = Layout::Clustered;
    variants[1].column_count = 2;
    variants[2].group_count = u64::MAX;
    variants[2].row_count = u64::MAX;
    variants[2].rows = Some(RowBounds {
        min: i64::MIN,
        max: i64::MAX,
    });
    for shape in variants {
        assert!(PayloadWriter::new(
            &mut sink,
            header(),
            shape,
            None,
            &SPEC,
            limits(),
            ColumnLimits::default(),
            &mut entries
        )
        .is_err());
        assert!(sink.is_empty());
    }
    let mut h = header();
    h.required_features |= REQUIRED_LEGACY_BASE;
    assert!(PayloadWriter::new(
        &mut sink,
        h,
        shape(1),
        None,
        &SPEC,
        limits(),
        ColumnLimits::default(),
        &mut entries
    )
    .is_err());
    let base = LegacyBase {
        generation: NonZeroU64::new(4).unwrap(),
        barrier_lsn: 100,
    };
    let wrong =
        PlannedCheckpoint::assert_captured_checkpoint(FileIdentity::new(1, 2, 99).unwrap(), base);
    assert!(PayloadWriter::new(
        &mut sink,
        h,
        shape(1),
        Some(&wrong),
        &SPEC,
        limits(),
        ColumnLimits::default(),
        &mut entries
    )
    .is_err());
    assert!(sink.is_empty());
}

#[test]
fn zero_column_groups_keep_complete_identity_and_directory_coverage() {
    for groups in [0, 1, 65] {
        let shape = VolumeShape {
            column_count: 0,
            ..shape(groups)
        };
        let mut file = Vec::new();
        let mut entries = Entries::default();
        let mut writer = PayloadWriter::new(
            &mut file,
            header(),
            shape,
            None,
            &[],
            limits(),
            ColumnLimits::default(),
            &mut entries,
        )
        .unwrap();
        let mut encoding = [0; 128];
        for group in 0..groups {
            let record = GroupRecord {
                column_count: 0,
                ..record(group)
            };
            writer
                .begin_group(record, &[group as i64], &[source()], &mut encoding)
                .unwrap();
            assert!(matches!(
                writer.write_column(&[], ColumnInput::AllNull, &mut encoding),
                Err(PayloadError::Sequence)
            ));
        }
        let mut complete = writer.finish_payloads().unwrap();
        assert_eq!(complete.descriptor_count(), 3 * groups);
        entries
            .0
            .sort_unstable_by(|a, b| a.key.cmp_identity(&b.key));
        let mut iter = entries.0.iter().copied();
        let finished = complete
            .finish(
                || Ok(iter.next()),
                &mut DirectoryScratch::new(),
                &mut [0; ENCODING_BYTES],
            )
            .unwrap();
        let disk = Bytes(&file);
        let envelope = OpenedEnvelope::read(&disk, file.len() as u64, &limits()).unwrap();
        let mut root_bytes = [0; 128];
        PageReadPlan::for_root(&envelope.footer, &limits())
            .unwrap()
            .read_into(&disk, &mut root_bytes, &mut [])
            .unwrap();
        let root = RootSummary::decode(&root_bytes, &envelope.footer, &limits()).unwrap();
        assert_eq!(root, finished.summary);
        assert_eq!(root.shape(), shape);
        let context = SourceContext::bind(&envelope.header, &root, None).unwrap();
        let mut nodes = [[0; super::super::directory::MAX_NODE_BYTES]; 2];
        let mut stored_node = [0; super::super::directory::MAX_NODE_BYTES];
        let mut walker = DirectoryWalker::new(
            &disk,
            envelope.footer,
            root,
            limits(),
            &mut nodes,
            &mut stored_node,
        )
        .unwrap();
        for (ordinal, expected) in entries.0.iter().copied().enumerate() {
            let entry = walker.next_entry().unwrap().unwrap();
            assert_eq!(entry, expected);
            assert_eq!(entry.key, expected_key(shape, ordinal as u64));
            let group = GroupExpectation::new(
                &root,
                entry.key.ordinal,
                GroupRecord {
                    column_count: 0,
                    ..record(entry.key.ordinal)
                },
            )
            .unwrap();
            let plan = PageReadPlan::for_page(&envelope.footer, entry.page, &limits()).unwrap();
            let bytes = plan.read_into(&disk, &mut encoding, &mut []).unwrap();
            match entry.key.section {
                n if n == Section::RowIds as u16 => assert_eq!(
                    IdentityPageExpectation::new(group)
                        .decode_row_ids(bytes)
                        .unwrap()
                        .iter()
                        .next(),
                    Some(entry.key.ordinal as i64)
                ),
                n if n == Section::SourceLsns as u16 => assert_eq!(
                    IdentityPageExpectation::new(group)
                        .decode_sources(bytes, context)
                        .unwrap()
                        .iter()
                        .next(),
                    Some(source())
                ),
                n if n == Section::GroupMetadata as u16 => {
                    GroupPageExpectation::new(&root, entry.key.ordinal, 1)
                        .unwrap()
                        .decode(bytes)
                        .unwrap();
                }
                _ => panic!("unexpected column page"),
            }
        }
        assert!(walker.next_entry().unwrap().is_none());
        walker.finish_validation().unwrap();
    }
}

#[test]
fn invalid_inputs_do_not_append_and_incomplete_groups_never_finish() {
    let mut file = Vec::new();
    let mut entries = Entries::default();
    let mut writer = PayloadWriter::new(
        &mut file,
        header(),
        shape(1),
        None,
        &SPEC,
        limits(),
        ColumnLimits::default(),
        &mut entries,
    )
    .unwrap();
    let mut encoding = [0; 128];
    assert!(writer
        .begin_group(record(0), &[9], &[source()], &mut encoding)
        .is_err());
    assert!(writer
        .begin_group(record(0), &[0], &[RowSource::LegacyBase], &mut encoding)
        .is_err());
    assert_eq!(writer.position(), 64);
    writer
        .begin_group(record(0), &[0], &[source()], &mut encoding)
        .unwrap();
    let before = writer.position();
    assert!(writer
        .write_column(&[false], ColumnInput::F64(&[2.0]), &mut encoding)
        .is_err());
    assert!(writer
        .write_column(&[false], ColumnInput::I64(&[2]), &mut [])
        .is_err());
    assert_eq!(writer.position(), before);
    assert!(writer
        .begin_group(record(0), &[0], &[source()], &mut encoding)
        .is_err());
    assert!(matches!(
        writer.finish_payloads(),
        Err(PayloadError::Incomplete)
    ));
    assert_eq!(entries.0.len(), 3);
}

#[test]
fn logical_row_caps_and_exact_aggregate_bounds() {
    for rows in [4095, 4096, 4097] {
        let mut file = Vec::new();
        let mut entries = Entries::default();
        let shape = VolumeShape {
            row_count: rows,
            rows: Some(RowBounds {
                min: 0,
                max: rows as i64 - 1,
            }),
            ..shape(1)
        };
        let mut writer = PayloadWriter::new(
            &mut file,
            header(),
            shape,
            None,
            &SPEC,
            limits(),
            ColumnLimits::default(),
            &mut entries,
        )
        .unwrap();
        let ids: Vec<_> = (0..rows as i64).collect();
        let sources = vec![source(); rows as usize];
        let mut encoding = vec![0; MAX_DECODED_BYTES];
        let rec = GroupRecord {
            row_count: rows as u32,
            rows: shape.rows.unwrap(),
            ..record(0)
        };
        let begun = writer.begin_group(rec, &ids, &sources, &mut encoding);
        if rows > 4096 {
            assert!(begun.is_err());
            assert_eq!(writer.position(), 64);
            continue;
        }
        begun.unwrap();
        writer
            .write_column(
                &vec![false; rows as usize],
                ColumnInput::I64(&ids),
                &mut encoding,
            )
            .unwrap();
        assert!(writer.finish_payloads().is_ok());
    }
    // Each local group may fit the declared extrema without actually covering
    // both extrema; complete coverage must still reject the incorrect root.
    let mut file = Vec::new();
    let mut entries = Entries::default();
    let wrong = VolumeShape {
        rows: Some(RowBounds { min: 0, max: 10 }),
        ..shape(1)
    };
    let mut writer = PayloadWriter::new(
        &mut file,
        header(),
        wrong,
        None,
        &SPEC,
        limits(),
        ColumnLimits::default(),
        &mut entries,
    )
    .unwrap();
    let mut encoding = [0; 128];
    writer
        .begin_group(record(0), &[0], &[source()], &mut encoding)
        .unwrap();
    writer
        .write_column(&[false], ColumnInput::I64(&[5]), &mut encoding)
        .unwrap();
    assert!(matches!(
        writer.finish_payloads(),
        Err(PayloadError::Group(GroupError::Incomplete))
    ));
}

#[test]
fn compressed_typed_blocks_fit_stored_limits_and_raw_fallback_roundtrips() {
    for repeated in [true, false] {
        let mut random = 123456789u64;
        let input: Vec<u8> = (0..100_000)
            .map(|_| {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                if repeated {
                    b'a'
                } else {
                    random as u8
                }
            })
            .collect();
        let kind = if repeated {
            DataType::Text
        } else {
            DataType::Vector
        };
        let columns = [ColumnSpec {
            data_type: kind,
            vector_dimensions: None,
        }];
        let mut page_limits = limits();
        if repeated {
            page_limits.page_stored_bytes = ENCODING_BYTES as u64;
        }
        let mut file = Vec::new();
        let mut entries = Entries::default();
        let mut writer = PayloadWriter::new(
            &mut file,
            header(),
            shape(1),
            None,
            &columns,
            page_limits,
            ColumnLimits::default(),
            &mut entries,
        )
        .unwrap();
        let mut decoded = vec![0; input.len() + 256];
        let mut compressed = vec![0; lz4_flex::block::get_maximum_output_size(decoded.len())];
        writer
            .begin_group(record(0), &[0], &[source()], &mut decoded)
            .unwrap();
        let before = writer.position();
        let offsets = [(0, input.len() as u64)];
        let values = ColumnInput::Variable {
            data: &input,
            offsets: &offsets,
        };
        if repeated {
            assert!(writer.write_column(&[false], values, &mut decoded).is_err());
            assert_eq!(writer.position(), before);
        }
        let mut small = CompressTable::small();
        assert!(matches!(
            writer.write_column_compressed(&[false], values, &mut decoded, &mut [], &mut small),
            Err(PayloadError::Compression(CompressionError::OutputTooShort))
        ));
        assert!(matches!(
            writer.write_column_compressed(
                &[false],
                values,
                &mut decoded,
                &mut compressed,
                &mut small
            ),
            Err(PayloadError::Compression(
                CompressionError::LargeTableRequired
            ))
        ));
        assert!(matches!(small, CompressTable::Small(_)));
        assert_eq!(writer.position(), before);
        let mut table = CompressTable::large();
        writer
            .write_column_compressed(&[false], values, &mut decoded, &mut compressed, &mut table)
            .unwrap();
        let mut complete = writer.finish_payloads().unwrap();
        let column = entries
            .0
            .iter()
            .find(|entry| entry.key.section == Section::ColumnBlocks as u16)
            .copied()
            .unwrap();
        assert_eq!(
            column.page.codec,
            if repeated {
                Codec::Lz4Block
            } else {
                Codec::Raw
            }
        );
        assert!(column.page.stored_len <= page_limits.page_stored_bytes);
        if repeated {
            assert!(column.page.decoded_len > page_limits.page_stored_bytes);
        }
        entries
            .0
            .sort_unstable_by(|a, b| a.key.cmp_identity(&b.key));
        let mut iter = entries.0.iter().copied();
        let finished = complete
            .finish(
                || Ok(iter.next()),
                &mut DirectoryScratch::new(),
                &mut [0; ENCODING_BYTES],
            )
            .unwrap();
        let disk = Bytes(&file);
        let plan = PageReadPlan::for_page(&finished.footer, column.page, &page_limits).unwrap();
        let bytes = plan
            .read_into(&disk, &mut compressed, &mut decoded)
            .unwrap();
        let expected = GroupExpectation::new(&finished.summary, 0, record(0))
            .unwrap()
            .column(0, kind, None)
            .unwrap();
        let block = ColumnBlockRef::parse(bytes, expected, ColumnLimits::default()).unwrap();
        match block.cell(0).unwrap() {
            ColumnCell::Text(text) => assert_eq!(text.as_bytes(), input),
            ColumnCell::Vector(vector) => {
                assert_eq!(vector.len() * 4, input.len());
                for (value, bytes) in vector.iter().zip(input.as_chunks::<4>().0) {
                    assert_eq!(value.to_bits(), u32::from_le_bytes(*bytes));
                }
            }
            _ => panic!("wrong decoded value type"),
        }
    }
}

#[test]
fn staged_legacy_base_is_bound_into_final_root() {
    let mut h = header();
    h.required_features |= REQUIRED_LEGACY_BASE;
    let base = LegacyBase {
        generation: NonZeroU64::new(12).unwrap(),
        barrier_lsn: 0,
    };
    let planned = PlannedCheckpoint::assert_captured_checkpoint(h.identity, base);
    let mut file = Vec::new();
    let mut entries = Entries::default();
    let mut writer = PayloadWriter::new(
        &mut file,
        h,
        shape(1),
        Some(&planned),
        &SPEC,
        limits(),
        ColumnLimits::default(),
        &mut entries,
    )
    .unwrap();
    let mut encoding = [0; 128];
    writer
        .begin_group(record(0), &[0], &[RowSource::LegacyBase], &mut encoding)
        .unwrap();
    writer
        .write_column(&[true], ColumnInput::AllNull, &mut encoding)
        .unwrap();
    let mut complete = writer.finish_payloads().unwrap();
    entries
        .0
        .sort_unstable_by(|a, b| a.key.cmp_identity(&b.key));
    let mut iter = entries.0.into_iter();
    let result = complete
        .finish(
            || Ok(iter.next()),
            &mut DirectoryScratch::new(),
            &mut [0; ENCODING_BYTES],
        )
        .unwrap();
    assert_eq!(result.summary.legacy_base, Some(base));
    let proof = VerifiedLegacyBase::assert_verified_checkpoint(h.identity, base);
    assert!(SourceContext::bind(&h, &result.summary, Some(&proof)).is_ok());
}

#[test]
fn descriptor_corruption_and_stream_panics_poison_completion() {
    for attack in 0..6 {
        let mut file = Vec::new();
        let mut entries = Entries::default();
        let mut writer = PayloadWriter::new(
            &mut file,
            header(),
            shape(1),
            None,
            &SPEC,
            limits(),
            ColumnLimits::default(),
            &mut entries,
        )
        .unwrap();
        let mut encoding = [0; 128];
        writer
            .begin_group(record(0), &[0], &[source()], &mut encoding)
            .unwrap();
        writer
            .write_column(&[false], ColumnInput::I64(&[2]), &mut encoding)
            .unwrap();
        let mut complete = writer.finish_payloads().unwrap();
        entries
            .0
            .sort_unstable_by(|a, b| a.key.cmp_identity(&b.key));
        match attack {
            0 => {
                entries.0.remove(2);
            }
            1 => {
                entries.0.push(entries.0[3]);
            }
            2 => entries.0[0].key.column = 1,
            3 => entries.0[0].page.offset = u64::MAX,
            4 => entries.0[0].page.offset = 0,
            _ => (),
        }
        let mut iter = entries.0.into_iter();
        let mut scratch = DirectoryScratch::new();
        let mut output = [0; ENCODING_BYTES];
        let result = catch_unwind(AssertUnwindSafe(|| {
            complete.finish(
                || {
                    assert!(attack != 5, "spool panic");
                    Ok(iter.next())
                },
                &mut scratch,
                &mut output,
            )
        }));
        assert!(result.is_err() || result.unwrap().is_err());
        assert!(matches!(
            complete.finish(|| Ok(None), &mut scratch, &mut output),
            Err(PayloadError::Poisoned)
        ));
    }
}

#[test]
fn page_and_descriptor_errors_or_panics_are_sticky() {
    struct FailFile {
        left: usize,
        panic: bool,
    }
    impl Write for FailFile {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.left == 0 {
                assert!(!self.panic, "file panic");
                return Err(io::ErrorKind::WriteZero.into());
            }
            let n = self.left.min(bytes.len());
            self.left -= n;
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    struct FailEntries {
        left: usize,
        panic: bool,
    }
    impl DescriptorSink for FailEntries {
        fn push(&mut self, _: LeafEntry) -> std::result::Result<(), RunError> {
            if self.left == 0 {
                assert!(!self.panic, "descriptor panic");
                return Err(RunError::Io(io::ErrorKind::Other.into()));
            }
            self.left -= 1;
            Ok(())
        }
    }
    for (file_left, entries_left) in [(80, 100), (usize::MAX, 0), (usize::MAX, 3)] {
        for (panic, compress) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut file = FailFile {
                left: file_left,
                panic,
            };
            let mut entries = FailEntries {
                left: entries_left,
                panic,
            };
            let mut writer = PayloadWriter::new(
                &mut file,
                header(),
                shape(1),
                None,
                &SPEC,
                limits(),
                ColumnLimits::default(),
                &mut entries,
            )
            .unwrap();
            let mut encoding = [0; 128];
            let mut compressed = [0; 256];
            let mut table = CompressTable::small();
            let result = catch_unwind(AssertUnwindSafe(|| {
                writer.begin_group(record(0), &[0], &[source()], &mut encoding)?;
                if compress {
                    writer.write_column_compressed(
                        &[false],
                        ColumnInput::I64(&[2]),
                        &mut encoding,
                        &mut compressed,
                        &mut table,
                    )
                } else {
                    writer.write_column(&[false], ColumnInput::I64(&[2]), &mut encoding)
                }
            }));
            assert!(result.is_err() || result.unwrap().is_err());
            assert!(matches!(
                writer.write_column(&[false], ColumnInput::I64(&[2]), &mut encoding),
                Err(PayloadError::Poisoned)
            ));
            assert!(matches!(
                writer.finish_payloads(),
                Err(PayloadError::Poisoned)
            ));
        }
    }
}
