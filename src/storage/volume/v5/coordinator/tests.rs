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

use super::super::column_block::{ColumnBlockRef, ColumnCell};
use super::super::directory::{LeafEntry, Section};
use super::super::envelope::FileIdentity;
use super::super::metadata_runs::RunError;
use super::super::row_spool::runs::{merge_pass, RowRunWriter};
use super::super::row_spool::{CapturedRow, SpoolBinding, SpoolLimits, SpoolWriter};
use super::*;
use crate::core::Value;
use std::cell::Cell;
use std::io::{self, Cursor};
use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};

struct Bytes<'a>(&'a [u8]);
impl ReadAt for Bytes<'_> {
    fn read_at(&self, offset: u64, out: &mut [u8]) -> io::Result<usize> {
        let at = usize::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?;
        let tail = self.0.get(at..).ok_or(io::ErrorKind::UnexpectedEof)?;
        let n = out.len().min(tail.len());
        out[..n].copy_from_slice(&tail[..n]);
        Ok(n)
    }
}
fn header() -> Header {
    Header::new(FileIdentity::new(7, 8, 9).unwrap())
}
fn specs(types: &[DataType]) -> Vec<ColumnSpec> {
    types
        .iter()
        .map(|&data_type| ColumnSpec {
            data_type,
            vector_dimensions: None,
        })
        .collect()
}
fn config(rows: u32, group: usize) -> BuildConfig {
    BuildConfig {
        rows,
        group_decoded_bytes: group,
        columns: ColumnLimits {
            decoded_bytes: 8192,
            ..ColumnLimits::default()
        },
        pages: ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: 8192,
            page_decoded_bytes: 8192,
        },
    }
}
fn fixture<'a>(
    columns: &'a [ColumnSpec],
    rows: &[(i64, Vec<Value>)],
) -> (Vec<u8>, Vec<u8>, RowRuns<'a>) {
    let binding = SpoolBinding {
        identity: header().identity,
        schema_version: NonZeroU64::new(11).unwrap(),
        columns,
        source: SourceEncodingContext::for_staged_file(&header(), None).unwrap(),
    };
    let mut payload = Cursor::new(Vec::new());
    let mut first = Cursor::new(Vec::new());
    let mut second = Cursor::new(Vec::new());
    let mut keys = [RowPosition::EMPTY; 3];
    let (mut w, mut r) = ([0; 128], [0; 128]);
    let mut writer =
        SpoolWriter::new(&mut payload, binding, SpoolLimits::default(), &mut w).unwrap();
    let mut runs = RowRunWriter::new(
        &mut first,
        binding,
        SpoolLimits::default(),
        &mut keys,
        &mut r,
    )
    .unwrap();
    for (i, (id, values)) in rows.iter().enumerate() {
        runs.push(
            writer
                .append(CapturedRow {
                    row_id: *id,
                    creator_txn_id: i as i64 + 1,
                    source: RowSource::Dml(NonZeroU64::new(i as u64 + 20).unwrap()),
                    values,
                })
                .unwrap(),
        )
        .unwrap();
    }
    let mut runs = runs.finish(writer.finish().unwrap()).unwrap();
    let (mut a, mut b, mut c) = ([0; 128], [0; 128], [0; 128]);
    while !runs.is_sorted() {
        runs = merge_pass(
            &Bytes(first.get_ref()),
            &mut second,
            runs,
            &mut a,
            &mut b,
            &mut c,
        )
        .unwrap();
        std::mem::swap(&mut first, &mut second);
    }
    (payload.into_inner(), first.into_inner(), runs)
}
struct Scratch {
    positions: Vec<RowPosition>,
    costs: Vec<u64>,
    ids: Vec<i64>,
    sources: Vec<RowSource>,
    sorted: Vec<u8>,
    window: Vec<u8>,
    plan: Vec<u8>,
    encoding: Vec<u8>,
    spans: Vec<CellSpan>,
    nulls: Vec<bool>,
    ints: Vec<i64>,
    floats: Vec<f64>,
    bools: Vec<bool>,
    timestamps: Vec<DateTime<Utc>>,
    bytes: Vec<u8>,
    offsets: Vec<(u64, u64)>,
}
impl Scratch {
    fn new(rows: usize) -> Self {
        Self {
            positions: vec![RowPosition::EMPTY; rows],
            costs: vec![0; rows],
            ids: vec![0; rows],
            sources: vec![RowSource::Dml(NonZeroU64::MIN); rows],
            sorted: vec![0; 128],
            window: vec![0; 256],
            plan: vec![0; 128],
            encoding: vec![0; 8192],
            spans: vec![CellSpan::EMPTY; rows],
            nulls: vec![false; rows],
            ints: vec![0; rows],
            floats: vec![0.; rows],
            bools: vec![false; rows],
            timestamps: vec![DateTime::UNIX_EPOCH; rows],
            bytes: vec![0; 8192],
            offsets: vec![(0, 0); rows],
        }
    }
    fn planning(&mut self) -> PlanningScratch<'_> {
        PlanningScratch {
            positions: &mut self.positions,
            prefix_costs: &mut self.costs,
            sorted_io: &mut self.sorted,
            payload_window: &mut self.window,
            plan_output: &mut self.plan,
            column: ColumnScratch {
                spans: &mut self.spans,
                nulls: &mut self.nulls,
                integers: &mut self.ints,
                floats: &mut self.floats,
                booleans: &mut self.bools,
                timestamps: &mut self.timestamps,
                bytes: &mut self.bytes,
                offsets: &mut self.offsets,
            },
        }
    }
    fn emission(&mut self) -> EmissionScratch<'_> {
        EmissionScratch {
            positions: &mut self.positions,
            row_ids: &mut self.ids,
            sources: &mut self.sources,
            sorted_io: &mut self.sorted,
            payload_window: &mut self.window,
            plan_input: &mut self.plan,
            encoding: &mut self.encoding,
            compression: None,
            column: ColumnScratch {
                spans: &mut self.spans,
                nulls: &mut self.nulls,
                integers: &mut self.ints,
                floats: &mut self.floats,
                booleans: &mut self.bools,
                timestamps: &mut self.timestamps,
                bytes: &mut self.bytes,
                offsets: &mut self.offsets,
            },
        }
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
fn page<'a>(file: &'a [u8], e: &LeafEntry) -> &'a [u8] {
    let bytes = &file[e.page.offset as usize..(e.page.offset + e.page.stored_len) as usize];
    assert_eq!(crc32fast::hash(bytes), e.page.stored_checksum);
    assert_eq!(e.page.stored_len, e.page.decoded_len);
    bytes
}
fn planned(bytes: &[u8]) -> Vec<PlannedGroup> {
    bytes
        .as_chunks::<PLAN_RECORD_BYTES>()
        .0
        .iter()
        .map(|b| PlannedGroup::decode(b).unwrap())
        .collect()
}

#[test]
fn compression_reservations_fail_before_header_or_descriptor_io() {
    let columns = specs(&[DataType::Integer]);
    let (payload, sorted, runs) = fixture(&columns, &[(1, vec![Value::Integer(7)])]);
    let mut config = config(1, 8192);
    config.columns.decoded_bytes = 65_535;
    config.pages.page_stored_bytes = 65_535;
    config.pages.page_decoded_bytes = 65_535;
    let mut scratch = Scratch::new(1);
    scratch.encoding.resize(65_535, 0);
    let mut boundary = Cursor::new(Vec::new());
    let plan = plan_groups(
        &Bytes(&sorted),
        &Bytes(&payload),
        &mut boundary,
        runs,
        config,
        &mut scratch.planning(),
    )
    .unwrap();
    let capacity = CompressionPlan::new(config.columns.decoded_bytes, &config.pages)
        .unwrap()
        .output_capacity();
    let mut output = vec![0xa5; capacity];
    let mut table = CompressTable::small();
    for (len, expected) in [
        (capacity - 1, CompressionError::OutputTooShort),
        (capacity, CompressionError::LargeTableRequired),
    ] {
        let mut file = vec![7; 19];
        let mut entries = Entries::default();
        let mut emission = scratch.emission();
        emission.compression = Some(CompressionScratch {
            output: &mut output[..len],
            table: &mut table,
        });
        assert!(matches!(
            emit_payloads(&Bytes(&sorted), &Bytes(&payload), &Bytes(boundary.get_ref()),
                &mut file, header(), None, plan, &mut entries, &mut emission),
            Err(CoordinatorError::Compression(error)) if error == expected
        ));
        assert_eq!(file, [7; 19]);
        assert!(entries.0.is_empty());
        assert!(output.iter().all(|&byte| byte == 0xa5));
        assert!(matches!(table, CompressTable::Small(_)));
    }
}

#[test]
fn compressed_and_incompressible_columns_match_raw_payloads_exactly() {
    use super::super::envelope::Codec;
    let columns = specs(&[DataType::Vector]);
    let mut state = 0x1234_5678_9abc_def0u64;
    let random = (0..16_384)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f32::from_bits(state as u32)
        })
        .collect();
    let rows = [
        (1, vec![Value::vector(vec![0.; 16_384])]),
        (2, vec![Value::vector(random)]),
        (3, vec![Value::Null(DataType::Vector)]),
    ];
    let (payload, sorted, runs) = fixture(&columns, &rows);
    let mut config = config(1, 131_072);
    config.columns.decoded_bytes = 131_072;
    config.pages.page_stored_bytes = 131_072;
    config.pages.page_decoded_bytes = 131_072;
    let mut scratch = Scratch::new(1);
    scratch.encoding.resize(131_072, 0);
    scratch.bytes.resize(131_072, 0);
    let mut boundary = Cursor::new(Vec::new());
    let plan = plan_groups(
        &Bytes(&sorted),
        &Bytes(&payload),
        &mut boundary,
        runs,
        config,
        &mut scratch.planning(),
    )
    .unwrap();
    let mut output = vec![
        0;
        CompressionPlan::new(config.columns.decoded_bytes, &config.pages)
            .unwrap()
            .output_capacity()
    ];
    let mut table = CompressTable::large();
    let mut files = [Vec::new(), Vec::new()];
    let mut descriptors = [Entries::default(), Entries::default()];
    for compress in [false, true] {
        let mut emission = scratch.emission();
        if compress {
            emission.compression = Some(CompressionScratch {
                output: &mut output,
                table: &mut table,
            });
        }
        emit_payloads(
            &Bytes(&sorted),
            &Bytes(&payload),
            &Bytes(boundary.get_ref()),
            &mut files[usize::from(compress)],
            header(),
            None,
            plan,
            &mut descriptors[usize::from(compress)],
            &mut emission,
        )
        .unwrap();
    }
    assert_eq!(descriptors[0].0.len(), descriptors[1].0.len());
    let mut decoded = vec![0; 131_072];
    for (raw, compressed) in descriptors[0].0.iter().zip(&descriptors[1].0) {
        assert_eq!(raw.key, compressed.key);
        assert_eq!(raw.page.decoded_len, compressed.page.decoded_len);
        let expected = page(&files[0], raw);
        let bytes = &files[1][compressed.page.offset as usize
            ..(compressed.page.offset + compressed.page.stored_len) as usize];
        assert_eq!(crc32fast::hash(bytes), compressed.page.stored_checksum);
        match compressed.page.codec {
            Codec::Raw => assert_eq!(bytes, expected),
            Codec::Lz4Block => {
                let len = lz4_flex::block::decompress_into(bytes, &mut decoded).unwrap();
                assert_eq!(&decoded[..len], expected);
            }
        }
        if raw.key.section == Section::ColumnBlocks as u16 {
            match raw.key.ordinal {
                0 => assert_eq!(compressed.page.codec, Codec::Lz4Block),
                1 => assert_eq!(compressed.page.codec, Codec::Raw),
                _ => {}
            }
        }
    }
}

#[test]
fn exact_sum_cap_controls_groups_and_preserves_sorted_identity() {
    let columns = specs(&[DataType::Integer, DataType::Text]);
    let rows: Vec<_> = (0..33)
        .rev()
        .map(|i| (i - 16, vec![Value::Integer(i * 17), Value::from("λ🚲")]))
        .collect();
    let (payload, sorted, runs) = fixture(&columns, &rows);
    for cap in [300, 400, 800, 1024, 4096] {
        let mut scratch = Scratch::new(33);
        let mut output = Cursor::new(Vec::new());
        let plan = plan_groups(
            &Bytes(&sorted),
            &Bytes(&payload),
            &mut output,
            runs,
            config(33, cap),
            &mut scratch.planning(),
        )
        .unwrap();
        let groups = planned(output.get_ref());
        assert_eq!(groups.len() as u64, plan.shape().group_count);
        assert!(groups.iter().all(|g| g.decoded_bytes <= cap as u64));
        let mut file = Vec::new();
        let mut entries = Entries::default();
        let complete = emit_payloads(
            &Bytes(&sorted),
            &Bytes(&payload),
            &Bytes(output.get_ref()),
            &mut file,
            header(),
            None,
            plan,
            &mut entries,
            &mut scratch.emission(),
        )
        .unwrap();
        assert_eq!(complete.descriptor_count(), groups.len() as u64 * 5);
        let _ = complete;
        for g in groups {
            let emitted: Vec<_> = entries
                .0
                .iter()
                .filter(|e| e.key.ordinal == g.ordinal)
                .collect();
            assert_eq!(
                emitted.iter().map(|e| e.page.decoded_len).sum::<u64>(),
                g.decoded_bytes
            );
            for e in emitted {
                let bytes = page(&file, e);
                if e.key.section == Section::RowIds as u16 {
                    for (i, b) in bytes[32..].as_chunks::<8>().0.iter().enumerate() {
                        assert_eq!(
                            i64::from_le_bytes(*b),
                            g.record.row_start as i64 + i as i64 - 16
                        );
                    }
                } else if e.key.section == Section::SourceLsns as u16 {
                    for (i, b) in bytes[32..].as_chunks::<8>().0.iter().enumerate() {
                        assert_eq!(u64::from_le_bytes(*b), 52 - g.record.row_start - i as u64);
                    }
                } else if e.key.section == Section::ColumnBlocks as u16 {
                    let expected = ColumnExpectation {
                        identity: ColumnIdentity {
                            physical_column: e.key.column,
                            group: g.ordinal,
                            row_start: g.record.row_start,
                            row_count: g.record.row_count,
                            data_type: columns[e.key.column as usize].data_type,
                        },
                        vector_dimensions: None,
                    };
                    let block =
                        ColumnBlockRef::parse(bytes, expected, config(33, cap).columns).unwrap();
                    for i in 0..g.record.row_count as usize {
                        assert_eq!(
                            block.cell(i),
                            Some(if e.key.column == 0 {
                                ColumnCell::Integer((g.record.row_start as i64 + i as i64) * 17)
                            } else {
                                ColumnCell::Text("λ🚲")
                            })
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn typed_null_empty_nan_vector_and_wide_timestamps_survive_emission() {
    let columns = specs(&[
        DataType::Float,
        DataType::Timestamp,
        DataType::Text,
        DataType::Vector,
        DataType::Null,
    ]);
    let wide = DateTime::from_timestamp(20_000_000_000, 987_654_321).unwrap();
    let leap = DateTime::from_timestamp(59, 1_500_000_123).unwrap();
    let nan = f64::from_bits(0xfff8_0000_0000_1234);
    let rows = vec![
        (
            i64::MAX,
            columns.iter().map(|c| Value::Null(c.data_type)).collect(),
        ),
        (
            0,
            vec![
                Value::Float(-0.0),
                Value::Timestamp(leap),
                Value::from("λ"),
                Value::vector(vec![]),
                Value::Null(DataType::Null),
            ],
        ),
        (
            i64::MIN,
            vec![
                Value::Float(nan),
                Value::Timestamp(wide),
                Value::from(""),
                Value::vector(vec![f32::from_bits(0x7fc01234), -0.0]),
                Value::Null(DataType::Null),
            ],
        ),
    ];
    let (payload, sorted, runs) = fixture(&columns, &rows);
    let mut scratch = Scratch::new(3);
    let mut boundaries = Cursor::new(Vec::new());
    let plan = plan_groups(
        &Bytes(&sorted),
        &Bytes(&payload),
        &mut boundaries,
        runs,
        config(3, 4096),
        &mut scratch.planning(),
    )
    .unwrap();
    assert_eq!(plan.shape.group_count, 1);
    let mut file = Vec::new();
    let mut entries = Entries::default();
    let _ = emit_payloads(
        &Bytes(&sorted),
        &Bytes(&payload),
        &Bytes(boundaries.get_ref()),
        &mut file,
        header(),
        None,
        plan,
        &mut entries,
        &mut scratch.emission(),
    )
    .unwrap();
    for e in entries
        .0
        .iter()
        .filter(|e| e.key.section == Section::ColumnBlocks as u16)
    {
        let expected = ColumnExpectation {
            identity: ColumnIdentity {
                physical_column: e.key.column,
                group: 0,
                row_start: 0,
                row_count: 3,
                data_type: columns[e.key.column as usize].data_type,
            },
            vector_dimensions: None,
        };
        let block =
            ColumnBlockRef::parse(page(&file, e), expected, config(3, 4096).columns).unwrap();
        assert_eq!(
            block.cell(2),
            Some(ColumnCell::Null(expected.identity.data_type))
        );
        match e.key.column {
            0 => {
                let Some(ColumnCell::Float(a)) = block.cell(0) else {
                    panic!()
                };
                let Some(ColumnCell::Float(b)) = block.cell(1) else {
                    panic!()
                };
                assert_eq!(a.to_bits(), nan.to_bits());
                assert_eq!(b.to_bits(), (-0.0f64).to_bits());
            }
            1 => {
                assert_eq!(
                    block.cell(0),
                    Some(ColumnCell::TimestampParts {
                        seconds: wide.timestamp(),
                        subsec_nanos: wide.timestamp_subsec_nanos()
                    })
                );
                assert_eq!(
                    block.cell(1),
                    Some(ColumnCell::TimestampParts {
                        seconds: leap.timestamp(),
                        subsec_nanos: leap.timestamp_subsec_nanos()
                    })
                );
            }
            2 => {
                assert_eq!(block.cell(0), Some(ColumnCell::Text("")));
                assert_eq!(block.cell(1), Some(ColumnCell::Text("λ")));
            }
            3 => {
                let Some(ColumnCell::Vector(v)) = block.cell(0) else {
                    panic!()
                };
                assert_eq!(
                    v.iter().map(f32::to_bits).collect::<Vec<_>>(),
                    [0x7fc01234, 0x80000000]
                );
                let Some(ColumnCell::Vector(v)) = block.cell(1) else {
                    panic!()
                };
                assert_eq!(v.len(), 0);
            }
            4 => assert_eq!(block.cell(0), Some(ColumnCell::Null(DataType::Null))),
            _ => unreachable!(),
        }
    }
}

#[test]
fn empty_spool_zero_columns_and_minimum_schema_cap_are_explicit() {
    for count in [0, 9] {
        let rows: Vec<_> = (0..count).rev().map(|i| (i, vec![])).collect();
        let (payload, sorted, runs) = fixture(&[], &rows);
        let mut scratch = Scratch::new(9);
        let mut boundaries = Cursor::new(Vec::new());
        let plan = plan_groups(
            &Bytes(&sorted),
            &Bytes(&payload),
            &mut boundaries,
            runs,
            config(9, 256),
            &mut scratch.planning(),
        )
        .unwrap();
        let mut file = Vec::new();
        let mut entries = Entries::default();
        let _ = emit_payloads(
            &Bytes(&sorted),
            &Bytes(&payload),
            &Bytes(boundaries.get_ref()),
            &mut file,
            header(),
            None,
            plan,
            &mut entries,
            &mut scratch.emission(),
        )
        .unwrap();
        assert_eq!(plan.shape.row_count, count as u64);
        assert_eq!(entries.0.len() as u64, plan.shape.group_count * 3);
        assert!(planned(boundaries.get_ref())
            .iter()
            .all(|g| g.decoded_bytes == 128 + 16 * u64::from(g.record.row_count)));
    }
    let columns = specs(&[DataType::Null; 20]);
    let (payload, sorted, runs) = fixture(&columns, &[(1, vec![Value::Null(DataType::Null); 20])]);
    let mut scratch = Scratch::new(1);
    let mut out = Cursor::new(vec![7; 100]);
    assert!(matches!(
        plan_groups(
            &Bytes(&sorted),
            &Bytes(&payload),
            &mut out,
            runs,
            config(1, 1000),
            &mut scratch.planning()
        ),
        Err(CoordinatorError::MinimumGroupBytes { required: 1444, .. })
    ));
    assert_eq!(out.into_inner(), vec![7; 100]);
}

#[test]
fn oversized_existing_cell_is_reported_before_any_plan_or_v5_publication() {
    let columns = specs(&[DataType::Text]);
    let (payload, sorted, runs) = fixture(&columns, &[(4, vec![Value::from("x".repeat(3000))])]);
    let original = payload.clone();
    let mut scratch = Scratch::new(1);
    let mut out = Cursor::new(vec![7; 100]);
    assert!(matches!(
        plan_groups(
            &Bytes(&sorted),
            &Bytes(&payload),
            &mut out,
            runs,
            config(1, 1000),
            &mut scratch.planning()
        ),
        Err(CoordinatorError::CannotFitRow { row_id: 4, .. })
    ));
    assert_eq!(out.into_inner(), vec![7; 100]);
    assert_eq!(payload, original);
    scratch.positions.clear();
    let mut out = Cursor::new(vec![9; 100]);
    assert!(matches!(
        plan_groups(
            &Bytes(&sorted),
            &Bytes(&payload),
            &mut out,
            runs,
            config(1, 4096),
            &mut scratch.planning()
        ),
        Err(CoordinatorError::BufferTooSmall)
    ));
    assert_eq!(out.into_inner(), vec![9; 100]);
}

#[test]
fn checked_plan_rejects_corruption_substitution_and_short_io_ignores_stale_tail() {
    let columns = specs(&[DataType::Integer]);
    let rows: Vec<_> = (0..9).rev().map(|i| (i, vec![Value::Integer(i)])).collect();
    let (payload, sorted, runs) = fixture(&columns, &rows);
    let mut scratch = Scratch::new(9);
    let mut boundaries = Cursor::new(Vec::new());
    let plan = plan_groups(
        &Bytes(&sorted),
        &Bytes(&payload),
        &mut boundaries,
        runs,
        config(9, 350),
        &mut scratch.planning(),
    )
    .unwrap();
    let original = boundaries.into_inner();
    for index in [0, 8, 16, 24, 28, 32, 40, 48, 56, 60] {
        let mut damaged = original.clone();
        damaged[index] ^= 1;
        // Also exercise a well-checksummed record whose aggregate stream receipt differs.
        for repair in [false, true]
            .into_iter()
            .filter(|repair| !*repair || index != 60)
        {
            if repair {
                let crc = crc32fast::hash(&damaged[..60]);
                damaged[60..64].copy_from_slice(&crc.to_le_bytes());
            }
            let mut out = Vec::new();
            let mut entries = Entries::default();
            assert!(emit_payloads(
                &Bytes(&sorted),
                &Bytes(&payload),
                &Bytes(&damaged),
                &mut out,
                header(),
                None,
                plan,
                &mut entries,
                &mut scratch.emission()
            )
            .is_err());
        }
    }
    let mut stale = original.clone();
    stale.extend_from_slice(&[0xff; 137]);
    let mut out = Vec::new();
    let mut entries = Entries::default();
    let _ = emit_payloads(
        &Bytes(&sorted),
        &Bytes(&payload),
        &Bytes(&stale),
        &mut out,
        header(),
        None,
        plan,
        &mut entries,
        &mut scratch.emission(),
    )
    .unwrap();
    let mut out = vec![];
    let mut entries = Entries::default();
    assert!(matches!(
        emit_payloads(
            &Bytes(&sorted),
            &Bytes(&payload),
            &Bytes(&original),
            &mut out,
            Header::new(FileIdentity::new(7, 8, 10).unwrap()),
            None,
            plan,
            &mut entries,
            &mut scratch.emission()
        ),
        Err(CoordinatorError::Identity)
    ));
    assert!(out.is_empty());
    assert!(entries.0.is_empty());
    let mut short = Short {
        bytes: &original,
        calls: Cell::new(0),
        lie: false,
    };
    let _ = emit_payloads(
        &Bytes(&sorted),
        &Bytes(&payload),
        &short,
        &mut out,
        header(),
        None,
        plan,
        &mut entries,
        &mut scratch.emission(),
    )
    .unwrap();
    assert!(short.calls.get() > original.len() / 3);
    short.lie = true;
    let mut out = vec![];
    let mut entries = Entries::default();
    assert!(emit_payloads(
        &Bytes(&sorted),
        &Bytes(&payload),
        &short,
        &mut out,
        header(),
        None,
        plan,
        &mut entries,
        &mut scratch.emission()
    )
    .is_err());
    for len in [0, 1, 63, original.len() - 1] {
        let mut out = vec![];
        let mut entries = Entries::default();
        assert!(emit_payloads(
            &Bytes(&sorted),
            &Bytes(&payload),
            &Bytes(&original[..len]),
            &mut out,
            header(),
            None,
            plan,
            &mut entries,
            &mut scratch.emission()
        )
        .is_err());
    }
}
struct Short<'a> {
    bytes: &'a [u8],
    calls: Cell<usize>,
    lie: bool,
}
impl ReadAt for Short<'_> {
    fn read_at(&self, at: u64, out: &mut [u8]) -> io::Result<usize> {
        let n = self.calls.get();
        self.calls.set(n + 1);
        if self.lie {
            return Ok(out.len() + 1);
        }
        if n.is_multiple_of(4) {
            return Err(io::ErrorKind::Interrupted.into());
        }
        let len = out.len().min(3);
        Bytes(self.bytes).read_at(at, &mut out[..len])
    }
}
struct Failing {
    sink: Cursor<Vec<u8>>,
    remaining: usize,
    panic: bool,
    lie: bool,
}
impl Write for Failing {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            if self.panic {
                panic!("injected write panic")
            }
            return Err(io::ErrorKind::Other.into());
        }
        if self.lie {
            return Ok(b.len() + 1);
        }
        let n = b.len().min(self.remaining);
        self.remaining -= n;
        self.sink.write(&b[..n])
    }
    fn flush(&mut self) -> io::Result<()> {
        panic!("coordinator must not flush")
    }
}
impl Seek for Failing {
    fn seek(&mut self, p: SeekFrom) -> io::Result<u64> {
        self.sink.seek(p)
    }
}
#[test]
fn partial_write_error_or_unwind_never_returns_a_finished_capability() {
    let columns = specs(&[DataType::Integer]);
    let (payload, sorted, runs) = fixture(&columns, &[(1, vec![Value::Integer(9)])]);
    let mut scratch = Scratch::new(1);
    for panic in [false, true] {
        let mut sink = Failing {
            sink: Cursor::new(Vec::new()),
            remaining: 7,
            panic,
            lie: false,
        };
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            plan_groups(
                &Bytes(&sorted),
                &Bytes(&payload),
                &mut sink,
                runs,
                config(1, 512),
                &mut scratch.planning(),
            )
        }));
        if panic {
            assert!(outcome.is_err());
        } else {
            assert!(outcome.unwrap().is_err());
        }
        assert_eq!(sink.sink.get_ref().len(), 7);
    }
    let mut lying = Failing {
        sink: Cursor::new(Vec::new()),
        remaining: 100,
        panic: false,
        lie: true,
    };
    assert!(plan_groups(
        &Bytes(&sorted),
        &Bytes(&payload),
        &mut lying,
        runs,
        config(1, 512),
        &mut scratch.planning()
    )
    .is_err());
    let mut boundaries = Cursor::new(Vec::new());
    let plan = plan_groups(
        &Bytes(&sorted),
        &Bytes(&payload),
        &mut boundaries,
        runs,
        config(1, 512),
        &mut scratch.planning(),
    )
    .unwrap();
    for (panic, lie) in [(false, false), (true, false)] {
        let mut sink = Failing {
            sink: Cursor::new(Vec::new()),
            remaining: 70,
            panic,
            lie,
        };
        let mut entries = Entries::default();
        let outcome = catch_unwind(AssertUnwindSafe(|| {
            emit_payloads(
                &Bytes(&sorted),
                &Bytes(&payload),
                &Bytes(boundaries.get_ref()),
                &mut sink,
                header(),
                None,
                plan,
                &mut entries,
                &mut scratch.emission(),
            )
            .map(drop)
        }));
        if panic {
            assert!(outcome.is_err());
        } else {
            assert!(outcome.unwrap().is_err());
        }
    }
}

struct Counted<'a> {
    bytes: &'a [u8],
    calls: Cell<usize>,
    read: Cell<usize>,
}
impl ReadAt for Counted<'_> {
    fn read_at(&self, offset: u64, out: &mut [u8]) -> io::Result<usize> {
        self.calls.set(self.calls.get() + 1);
        let n = Bytes(self.bytes).read_at(offset, out)?;
        self.read.set(self.read.get() + n);
        Ok(n)
    }
}
#[test]
fn late_large_column_planning_reads_scale_with_rows_not_suffixes() {
    let mut columns = specs(&[DataType::Boolean; 16]);
    columns.extend(specs(&[DataType::Text]));
    let mut previous = 0;
    for count in [512, 1024, 4096] {
        let text = Value::from("x".repeat(6000));
        let rows: Vec<_> = (0..count)
            .rev()
            .map(|i| {
                let mut values = vec![Value::Boolean(i % 2 == 0); 16];
                values.push(text.clone());
                (i as i64, values)
            })
            .collect();
        let (payload, sorted, runs) = fixture(&columns, &rows);
        let source = Counted {
            bytes: &payload,
            calls: Cell::new(0),
            read: Cell::new(0),
        };
        let mut scratch = Scratch::new(4096);
        scratch.window.resize(8192, 0);
        let mut out = Cursor::new(Vec::new());
        let plan = plan_groups(
            &Bytes(&sorted),
            &source,
            &mut out,
            runs,
            config(4096, 8000),
            &mut scratch.planning(),
        )
        .unwrap();
        assert_eq!(plan.shape.group_count, count as u64); // The last column limits every group to one row.
        assert!(
            source.calls.get() < 32 * count,
            "{} reads for {count} rows",
            source.calls.get()
        );
        if count == 1024 {
            assert!(source.calls.get() <= previous * 2 + 32);
        }
        previous = source.calls.get();
        eprintln!(
            "late large column: {count} rows, {} reads / {} bytes, window8192",
            source.calls.get(),
            source.read.get()
        );
    }
}
#[test]
fn rounded_null_maps_pack_wide_columns_and_timestamp_fallback_is_exact() {
    for ty in [DataType::Null, DataType::Boolean] {
        let columns = specs(&[ty; 40]);
        let rows: Vec<_> = (0..33)
            .rev()
            .map(|i| {
                (
                    i,
                    columns
                        .iter()
                        .map(|_| {
                            if ty == DataType::Null {
                                Value::Null(ty)
                            } else {
                                Value::Boolean(i % 2 == 0)
                            }
                        })
                        .collect(),
                )
            })
            .collect();
        let (payload, sorted, runs) = fixture(&columns, &rows);
        let cap = 128
            + 16 * 33
            + 40 * (64 + 33usize.div_ceil(8) + if ty == DataType::Boolean { 33 } else { 0 });
        let mut scratch = Scratch::new(33);
        let mut out = Cursor::new(Vec::new());
        let plan = plan_groups(
            &Bytes(&sorted),
            &Bytes(&payload),
            &mut out,
            runs,
            config(33, cap),
            &mut scratch.planning(),
        )
        .unwrap();
        assert_eq!(plan.shape.group_count, 1);
        assert_eq!(planned(out.get_ref())[0].decoded_bytes, cap as u64);
    }
    let columns = specs(&[DataType::Timestamp]);
    for timestamp in [
        DateTime::UNIX_EPOCH,
        DateTime::from_timestamp(20_000_000_000, 3).unwrap(),
        DateTime::from_timestamp(59, 1_500_000_000).unwrap(),
    ] {
        let (payload, sorted, runs) = fixture(&columns, &[(3, vec![Value::Timestamp(timestamp)])]);
        let narrow = timestamp == DateTime::UNIX_EPOCH;
        let cap = 144 + 65 + if narrow { 8 } else { 12 };
        let mut scratch = Scratch::new(1);
        let mut out = Cursor::new(Vec::new());
        let plan = plan_groups(
            &Bytes(&sorted),
            &Bytes(&payload),
            &mut out,
            runs,
            config(1, cap),
            &mut scratch.planning(),
        )
        .unwrap();
        assert_eq!(plan.shape.group_count, 1);
        assert_eq!(planned(out.get_ref())[0].decoded_bytes, cap as u64);
    }
}

#[test]
fn source_context_and_same_width_unrelated_schema_cannot_replace_the_bound_input() {
    use super::super::envelope::{LegacyBase, REQUIRED_LEGACY_BASE};
    let columns = specs(&[DataType::Integer]);
    let unrelated = specs(&[DataType::Float]);
    let (payload, sorted, runs) = fixture(&columns, &[(1, vec![Value::Integer(7)])]);
    let (wrong_payload, _, _) = fixture(&unrelated, &[(1, vec![Value::Float(7.)])]);
    let mut scratch = Scratch::new(1);
    let mut boundary = Cursor::new(Vec::new());
    let plan = plan_groups(
        &Bytes(&sorted),
        &Bytes(&payload),
        &mut boundary,
        runs,
        config(1, 512),
        &mut scratch.planning(),
    )
    .unwrap();
    let mut out = Vec::new();
    let mut entries = Entries::default();
    assert!(emit_payloads(
        &Bytes(&sorted),
        &Bytes(&wrong_payload),
        &Bytes(boundary.get_ref()),
        &mut out,
        header(),
        None,
        plan,
        &mut entries,
        &mut scratch.emission()
    )
    .is_err());
    let mut h = header();
    h.required_features |= REQUIRED_LEGACY_BASE;
    let checkpoint = PlannedCheckpoint::assert_captured_checkpoint(
        h.identity,
        LegacyBase {
            generation: NonZeroU64::new(17).unwrap(),
            barrier_lsn: 9,
        },
    );
    for checkpoint in [None, Some(&checkpoint)] {
        let mut out = Vec::new();
        let mut entries = Entries::default();
        assert!(matches!(
            emit_payloads(
                &Bytes(&sorted),
                &Bytes(&payload),
                &Bytes(boundary.get_ref()),
                &mut out,
                h,
                checkpoint,
                plan,
                &mut entries,
                &mut scratch.emission()
            ),
            Err(CoordinatorError::Identity)
        ));
        assert!(out.is_empty());
        assert!(entries.0.is_empty());
    }
}
