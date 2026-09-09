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

use super::super::envelope::{Header, LegacyBase, REQUIRED_LEGACY_BASE};
use super::super::row_identity::PlannedCheckpoint;
use super::runs::*;
use super::*;
use std::cell::Cell;
use std::io::Cursor;

struct Bytes<'a> {
    bytes: &'a [u8],
    calls: Cell<usize>,
    read: Cell<usize>,
    max: usize,
    fail: Option<usize>,
    interrupt: Cell<bool>,
    lie: bool,
}
impl<'a> Bytes<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            calls: Cell::new(0),
            read: Cell::new(0),
            max: usize::MAX,
            fail: None,
            interrupt: Cell::new(false),
            lie: false,
        }
    }
}
impl ReadAt for Bytes<'_> {
    fn read_at(&self, offset: u64, out: &mut [u8]) -> io::Result<usize> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if self.interrupt.replace(false) {
            return Err(io::ErrorKind::Interrupted.into());
        }
        if self.lie {
            return Ok(out.len() + 1);
        }
        if self.fail.is_some_and(|limit| call >= limit) {
            return Err(io::ErrorKind::Other.into());
        }
        let offset = usize::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?;
        let tail = self
            .bytes
            .get(offset..)
            .ok_or(io::ErrorKind::UnexpectedEof)?;
        let n = out.len().min(tail.len()).min(self.max);
        out[..n].copy_from_slice(&tail[..n]);
        self.read.set(self.read.get() + n);
        Ok(n)
    }
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
fn binding(columns: &[ColumnSpec]) -> SpoolBinding<'_> {
    let identity = FileIdentity::new(2, 3, 4).unwrap();
    SpoolBinding {
        identity,
        schema_version: NonZeroU64::new(9).unwrap(),
        columns,
        source: SourceEncodingContext::for_staged_file(&Header::new(identity), None).unwrap(),
    }
}
fn captured(id: i64, values: &[Value]) -> CapturedRow<'_> {
    CapturedRow {
        row_id: id,
        creator_txn_id: 17,
        source: RowSource::Dml(NonZeroU64::new(31).unwrap()),
        values,
    }
}
fn fixture<'a>(
    columns: &'a [ColumnSpec],
    rows: &[(i64, Vec<Value>)],
) -> (Vec<u8>, Vec<RowPosition>, FinishedSpool<'a>) {
    let mut sink = Cursor::new(Vec::new());
    let mut buffer = [0; 128];
    let mut writer = SpoolWriter::new(
        &mut sink,
        binding(columns),
        SpoolLimits::default(),
        &mut buffer,
    )
    .unwrap();
    let positions = rows
        .iter()
        .map(|(id, values)| writer.append(captured(*id, values)).unwrap())
        .collect();
    let summary = writer.finish().unwrap();
    (sink.into_inner(), positions, summary)
}
fn sorted<'a>(
    columns: &'a [ColumnSpec],
    ids: &[i64],
) -> Result<(Vec<u8>, Vec<RowPosition>, FinishedSpool<'a>)> {
    let bound = binding(columns);
    let limits = SpoolLimits::default();
    let mut payload = Cursor::new(Vec::new());
    let mut first = Cursor::new(Vec::new());
    let mut second = Cursor::new(Vec::new());
    let mut keys = [RowPosition::EMPTY; 2];
    let mut write_io = [0; 128];
    let mut run_io = [0; 128];
    let mut writer = SpoolWriter::new(&mut payload, bound, limits, &mut write_io)?;
    let mut runs = RowRunWriter::new(&mut first, bound, limits, &mut keys, &mut run_io)?;
    for &id in ids {
        let values = [Value::Integer(id)];
        runs.push(writer.append(captured(id, &values))?)?;
    }
    let summary = writer.finish()?;
    let mut meta = runs.finish(summary)?;
    let (mut a, mut b, mut c) = ([0; 128], [0; 128], [0; 128]);
    while !meta.is_sorted() {
        meta = merge_pass(
            &Bytes::new(first.get_ref()),
            &mut second,
            meta,
            &mut a,
            &mut b,
            &mut c,
        )?;
        std::mem::swap(&mut first, &mut second);
    }
    first.get_mut().extend_from_slice(&[0xff; 300]); // Stale tail is outside opaque logical length.
    let source = Bytes::new(first.get_ref());
    let mut reader = SortedRows::new(&source, meta, &mut a)?;
    let mut output = Vec::new();
    while let Some(row) = reader.next_position()? {
        output.push(row);
    }
    assert_eq!(source.read.get() as u64, meta.byte_len());
    Ok((payload.into_inner(), output, summary))
}

#[test]
fn external_sort_retains_signed_identity_payload_and_detects_cross_run_duplicates() {
    let columns = specs(&[DataType::Integer]);
    for ids in [
        vec![],
        vec![i64::MAX, 0, -3, i64::MIN, 8, -9, 1],
        (0..257).rev().collect(),
    ] {
        let (bytes, rows, summary) = sorted(&columns, &ids).unwrap();
        assert_eq!(rows.len(), ids.len());
        assert!(rows.windows(2).all(|v| v[0].row_id() < v[1].row_id()));
        if rows.is_empty() {
            continue;
        }
        let mut window = [0; 1024];
        let source = Bytes::new(&bytes);
        let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
        let mut spans = vec![CellSpan::EMPTY; rows.len()];
        let mut nulls = vec![true; rows.len()];
        let mut values = vec![0; rows.len()];
        let result = reader
            .gather_column(
                &rows,
                0,
                &mut spans,
                &mut nulls,
                ColumnBuffer::I64(&mut values),
            )
            .unwrap();
        let ColumnInput::I64(values) = result.input else {
            panic!()
        };
        for (row, value) in rows.iter().zip(values) {
            assert_eq!(*value, row.row_id());
            assert_eq!(row.creator_txn_id(), 17);
            assert_eq!(row.source(), RowSource::Dml(NonZeroU64::new(31).unwrap()));
        }
    }
    for ids in [vec![3, 1, 8, 3], vec![3, 3, 8, 9], vec![9, 8, 3, 2, 1, 9]] {
        assert!(matches!(sorted(&columns, &ids), Err(SpoolError::RowOrder)));
    }
}

#[test]
fn typed_gather_preserves_null_empty_float_vector_unicode_and_wide_leap_timestamps() {
    let columns = specs(&[
        DataType::Integer,
        DataType::Float,
        DataType::Boolean,
        DataType::Timestamp,
        DataType::Text,
        DataType::Json,
        DataType::Vector,
        DataType::Null,
    ]);
    let wide = DateTime::from_timestamp(20_000_000_000, 987_654_321).unwrap();
    let leap = DateTime::from_timestamp(59, 1_500_000_123).unwrap();
    let nan = f64::from_bits(0xfff8_0000_0000_1234);
    let vector = vec![f32::from_bits(0x7fc01234), -0.0, f32::INFINITY];
    let rows = vec![
        (
            -4,
            vec![
                Value::Integer(i64::MIN),
                Value::Float(nan),
                Value::Boolean(true),
                Value::Timestamp(wide),
                Value::from(""),
                Value::json("{\"ç\":1}"),
                Value::vector(vector.clone()),
                Value::Null(DataType::Null),
            ],
        ),
        (
            7,
            vec![
                Value::Integer(i64::MAX),
                Value::Float(-0.0),
                Value::Boolean(false),
                Value::Timestamp(leap),
                Value::from("λ🚲"),
                Value::json(""),
                Value::vector(vec![]),
                Value::Null(DataType::Null),
            ],
        ),
        (
            9,
            columns.iter().map(|s| Value::Null(s.data_type)).collect(),
        ),
    ];
    let (bytes, positions, summary) = fixture(&columns, &rows);
    let source = Bytes::new(&bytes);
    let mut window = [0; 128];
    let mut spans = [CellSpan::EMPTY; 3];
    let mut nulls = [false; 3];
    let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
    let mut group = reader.prepare_group(&positions).unwrap();
    let mut ints = [0; 3];
    let result = group
        .gather_column(0, &mut spans, &mut nulls, ColumnBuffer::I64(&mut ints))
        .unwrap();
    assert_eq!(result.nulls, [false, false, true]);
    assert_eq!(ints, [i64::MIN, i64::MAX, 0]);
    let mut floats = [0.0; 3];
    group
        .gather_column(1, &mut spans, &mut nulls, ColumnBuffer::F64(&mut floats))
        .unwrap();
    assert_eq!(floats[0].to_bits(), nan.to_bits());
    assert_eq!(floats[1].to_bits(), (-0.0f64).to_bits());
    let mut bools = [false; 3];
    group
        .gather_column(2, &mut spans, &mut nulls, ColumnBuffer::Bool(&mut bools))
        .unwrap();
    assert_eq!(bools, [true, false, false]);
    let mut stamps = [DateTime::UNIX_EPOCH; 3];
    group
        .gather_column(
            3,
            &mut spans,
            &mut nulls,
            ColumnBuffer::Timestamps(&mut stamps),
        )
        .unwrap();
    assert_eq!(stamps[..2], [wide, leap]);
    for (column, expected) in [
        (4, vec![b"".as_slice(), "λ🚲".as_bytes()]),
        (5, vec!["{\"ç\":1}".as_bytes(), b"".as_slice()]),
        (
            6,
            vec![
                match &rows[0].1[6] {
                    Value::Extension(bytes) => &bytes[1..],
                    _ => panic!(),
                },
                b"".as_slice(),
            ],
        ),
    ] {
        let mut bytes = [0; 128];
        let mut offsets = [(99, 99); 3];
        let result = group
            .gather_column(
                column,
                &mut spans,
                &mut nulls,
                ColumnBuffer::Variable {
                    bytes: &mut bytes,
                    offsets: &mut offsets,
                },
            )
            .unwrap();
        let ColumnInput::Variable { data, offsets } = result.input else {
            panic!()
        };
        assert_eq!(result.nulls, [false, false, true]);
        for (offset, expected) in offsets.iter().zip(expected) {
            assert_eq!(
                &data[offset.0 as usize..(offset.0 + offset.1) as usize],
                expected
            );
        }
    }
    group
        .gather_column(7, &mut spans, &mut nulls, ColumnBuffer::AllNull)
        .unwrap();
    assert_eq!(nulls, [true; 3]);
}

#[test]
fn capacity_preflight_preserves_output_and_oversized_cells_remain_in_spool() {
    let columns = specs(&[DataType::Text]);
    let (bytes, rows, summary) = fixture(
        &columns,
        &[
            (1, vec![Value::from("abc")]),
            (2, vec![Value::from("defg")]),
        ],
    );
    let source = Bytes::new(&bytes);
    let mut window = [0; 64];
    let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
    let mut group = reader.prepare_group(&rows).unwrap();
    let mut spans = [CellSpan::EMPTY; 2];
    let mut nulls = [true; 2];
    let mut out = [99; 6];
    let mut offsets = [(99, 99); 2];
    assert!(matches!(
        group.gather_column(
            0,
            &mut spans,
            &mut nulls,
            ColumnBuffer::Variable {
                bytes: &mut out,
                offsets: &mut offsets
            }
        ),
        Err(SpoolError::BufferTooSmall)
    ));
    assert_eq!(out, [99; 6]);
    assert_eq!(offsets, [(99, 99); 2]);
    assert_eq!(nulls, [true; 2]);
    let mut out = [0; 7];
    group
        .gather_column(
            0,
            &mut spans,
            &mut nulls,
            ColumnBuffer::Variable {
                bytes: &mut out,
                offsets: &mut offsets,
            },
        )
        .unwrap();
    assert_eq!(&out, b"abcdefg");
    let big = "x".repeat(MAX_DECODED_BYTES + 1);
    let (bytes, rows, summary) = fixture(&columns, &[(1, vec![Value::from(big.as_str())])]);
    assert!(summary.byte_len() > MAX_DECODED_BYTES as u64);
    let source = Bytes::new(&bytes);
    let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
    let mut group = reader.prepare_group(&rows).unwrap();
    let mut out = [99; 1];
    let mut offsets = [(99, 99)];
    let mut nulls = [true];
    assert!(matches!(
        group.gather_column(
            0,
            &mut spans,
            &mut nulls,
            ColumnBuffer::Variable {
                bytes: &mut out,
                offsets: &mut offsets
            }
        ),
        Err(SpoolError::OversizedCell)
    ));
    assert_eq!(out, [99]);
    assert_eq!(offsets, [(99, 99)]);
    assert_eq!(nulls, [true]);
}

#[test]
fn schema_identity_source_and_write_preflight_are_exact() {
    let columns = specs(&[DataType::Integer]);
    let bound = binding(&columns);
    let mut sink = Cursor::new(Vec::new());
    let mut io = [0; 64];
    let mut writer = SpoolWriter::new(
        &mut sink,
        bound,
        SpoolLimits {
            row_bytes: 104,
            cell_bytes: 8,
            ..SpoolLimits::default()
        },
        &mut io,
    )
    .unwrap();
    assert!(matches!(
        writer.append(captured(1, &[Value::from("wrong")])),
        Err(SpoolError::Type)
    ));
    assert!(matches!(
        writer.append(captured(1, &[])),
        Err(SpoolError::Schema)
    ));
    let mut row = captured(1, &[Value::Integer(1)]);
    row.creator_txn_id = 0;
    assert!(matches!(writer.append(row), Err(SpoolError::Identity)));
    let mut row = captured(1, &[Value::Integer(1)]);
    row.source = RowSource::LegacyBase;
    assert!(matches!(writer.append(row), Err(SpoolError::Source)));
    let receipt = writer.append(captured(1, &[Value::Integer(1)])).unwrap();
    let summary = writer.finish().unwrap();
    assert_eq!(summary.byte_len(), 104);
    assert_eq!(sink.get_ref().len(), 104);
    let other = specs(&[DataType::Float]);
    let mut run_file = Cursor::new(Vec::new());
    let mut keys = [RowPosition::EMPTY; 1];
    let mut run = RowRunWriter::new(
        &mut run_file,
        binding(&other),
        SpoolLimits::default(),
        &mut keys,
        &mut io,
    )
    .unwrap();
    run.push(receipt).unwrap();
    assert!(matches!(run.finish(summary), Err(SpoolError::Identity)));
    let base = LegacyBase {
        generation: NonZeroU64::new(5).unwrap(),
        barrier_lsn: 100,
    };
    let mut header = Header::new(bound.identity);
    header.required_features |= REQUIRED_LEGACY_BASE;
    let checkpoint = PlannedCheckpoint::assert_captured_checkpoint(bound.identity, base);
    let mut base_bound = bound;
    base_bound.source = SourceEncodingContext::for_staged_file(&header, Some(&checkpoint)).unwrap();
    assert_eq!(base_bound.source_lane(RowSource::LegacyBase).unwrap(), 0);
    assert!(matches!(
        base_bound.source_lane(RowSource::Dml(NonZeroU64::new(100).unwrap())),
        Err(SpoolError::Source)
    ));
    assert_eq!(
        base_bound
            .source_lane(RowSource::Dml(NonZeroU64::new(101).unwrap()))
            .unwrap(),
        101
    );
}

#[test]
fn corruption_headers_descriptors_payloads_and_partial_gather_poison_the_reader() {
    let columns = specs(&[DataType::Integer]);
    let (original, rows, summary) = fixture(
        &columns,
        &[(1, vec![Value::Integer(10)]), (2, vec![Value::Integer(20)])],
    );
    for offset in [0, HEADER_BYTES + 24, original.len() - 1] {
        let mut bytes = original.clone();
        bytes[offset] ^= 0x80;
        let source = Bytes::new(&bytes);
        let mut window = [0; 64];
        let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
        let mut spans = [CellSpan::EMPTY; 2];
        let mut nulls = [true; 2];
        let mut values = [999; 2];
        assert!(matches!(
            reader.gather_column(
                &rows,
                0,
                &mut spans,
                &mut nulls,
                ColumnBuffer::I64(&mut values)
            ),
            Err(SpoolError::Checksum)
        ));
        assert!(matches!(
            reader.gather_column(
                &rows,
                0,
                &mut spans,
                &mut nulls,
                ColumnBuffer::I64(&mut values)
            ),
            Err(SpoolError::Poisoned)
        ));
        if offset == original.len() - 1 {
            assert_eq!(values[0], 10);
        } // Partial output is explicitly invalid.
    }
}

#[test]
fn bounded_window_counts_contiguous_and_scattered_io_without_stale_tail_prefetch() {
    let columns = specs(&[DataType::Integer; 8]);
    let rows: Vec<_> = (0..128)
        .map(|i| (i, (0..8).map(|j| Value::Integer(i * 10 + j)).collect()))
        .collect();
    let (mut bytes, positions, summary) = fixture(&columns, &rows);
    bytes.extend_from_slice(&[99; 1024]);
    let source = Bytes::new(&bytes);
    let mut window = [0; MAX_IO_BYTES];
    let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
    let mut spans = [CellSpan::EMPTY; 128];
    let mut nulls = [false; 128];
    let mut values = [0; 128];
    for column in 0..8 {
        reader
            .gather_column(
                &positions,
                column,
                &mut spans,
                &mut nulls,
                ColumnBuffer::I64(&mut values),
            )
            .unwrap();
        for (i, value) in values.iter().enumerate() {
            assert_eq!(*value, i as i64 * 10 + i64::from(column));
        }
    }
    assert_eq!(source.calls.get(), 1);
    assert_eq!(source.read.get() as u64, summary.byte_len());
    let source = Bytes::new(&bytes);
    let mut small = [0; 512];
    let mut reader = SpoolReader::new(&source, summary, &mut small).unwrap();
    let mut shuffled = positions;
    shuffled.reverse();
    reader
        .gather_column(
            &shuffled,
            0,
            &mut spans,
            &mut nulls,
            ColumnBuffer::I64(&mut values),
        )
        .unwrap();
    assert!(source.calls.get() > 100); // Arbitrary payload positions genuinely scatter; no sequential-IO claim.
    for (i, value) in values.iter().enumerate() {
        assert_eq!(*value, (127 - i) as i64 * 10);
    }
}

struct WriterFault {
    inner: Cursor<Vec<u8>>,
    remaining: usize,
    lie: bool,
    interrupt: bool,
    panic: bool,
}
impl Seek for WriterFault {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}
impl Write for WriterFault {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.interrupt {
            self.interrupt = false;
            return Err(io::ErrorKind::Interrupted.into());
        }
        assert!(!self.panic, "injected write panic");
        if self.lie {
            return Ok(bytes.len() + 1);
        }
        if self.remaining == 0 {
            return Err(io::ErrorKind::Other.into());
        }
        let n = bytes.len().min(self.remaining).min(7);
        self.remaining -= n;
        self.inner.write(&bytes[..n])
    }
    fn flush(&mut self) -> io::Result<()> {
        panic!("spool must not flush the external sink")
    }
}
#[test]
fn short_interrupted_misreported_and_panicking_io_obeys_sticky_abort() {
    let columns = specs(&[DataType::Integer]);
    for (remaining, lie, panic) in [(12, false, false), (1000, true, false), (1000, false, true)] {
        let mut sink = WriterFault {
            inner: Cursor::new(Vec::new()),
            remaining,
            lie,
            interrupt: true,
            panic,
        };
        let mut io = [0; 64];
        let mut writer = SpoolWriter::new(
            &mut sink,
            binding(&columns),
            SpoolLimits::default(),
            &mut io,
        )
        .unwrap();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            writer.append(captured(1, &[Value::Integer(1)]))
        }));
        assert!(result.is_err() || result.unwrap().is_err());
        assert!(matches!(
            writer.append(captured(1, &[Value::Integer(1)])),
            Err(SpoolError::Poisoned)
        ));
    }
    let (bytes, rows, summary) = fixture(&columns, &[(1, vec![Value::Integer(99)])]);
    for (max, fail, lie) in [(7, None, false), (7, Some(4), false), (7, None, true)] {
        let mut source = Bytes::new(&bytes);
        source.max = max;
        source.fail = fail;
        source.lie = lie;
        source.interrupt.set(true);
        let mut window = [0; 64];
        let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
        let mut spans = [CellSpan::EMPTY];
        let mut nulls = [false];
        let mut values = [0];
        let result = reader.gather_column(
            &rows,
            0,
            &mut spans,
            &mut nulls,
            ColumnBuffer::I64(&mut values),
        );
        if fail.is_none() && !lie {
            assert!(result.is_ok());
            assert_eq!(values, [99]);
        } else {
            assert!(result.is_err());
            assert!(matches!(
                reader.gather_column(
                    &rows,
                    0,
                    &mut spans,
                    &mut nulls,
                    ColumnBuffer::I64(&mut values)
                ),
                Err(SpoolError::Poisoned)
            ));
        }
    }
}

// Recompute checksums deliberately: semantic validators must reject malformed
// tags/ranges even when a corrupt producer supplies self-consistent checksums.
fn resign_single_column(bytes: &mut [u8], position: &mut RowPosition) {
    let crc = crc32fast::hash(&bytes[64..92]);
    bytes[92..96].copy_from_slice(&crc.to_le_bytes());
    let crc = crc32fast::hash(&bytes[64..96]);
    bytes[48..52].copy_from_slice(&crc.to_le_bytes());
    let crc = crc32fast::hash(&bytes[..60]);
    bytes[60..64].copy_from_slice(&crc.to_le_bytes());
    position.header_crc = crc;
}
#[test]
fn correctly_checksummed_invalid_tags_ranges_and_scalar_payloads_fail_closed() {
    let integer = specs(&[DataType::Integer]);
    let (original, rows, summary) = fixture(&integer, &[(1, vec![Value::Integer(10)])]);
    for kind in 0..6 {
        let mut bytes = original.clone();
        let mut rows = rows.clone();
        match kind {
            0 => bytes[64 + 24] = DataType::Text as u8,
            1 => bytes[64 + 25] = 2,
            2 => bytes[64 + 8..64 + 16].copy_from_slice(&u64::MAX.to_le_bytes()),
            3 => bytes[64..64 + 8].copy_from_slice(&0u64.to_le_bytes()),
            4 => bytes[64 + 25] = 1, // NULL is not permitted to retain payload.
            5 => bytes[44] ^= 1,     // Same-width unrelated schema binding.
            _ => unreachable!(),
        }
        resign_single_column(&mut bytes, &mut rows[0]);
        let source = Bytes::new(&bytes);
        let mut window = [0; 128];
        let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
        let mut spans = [CellSpan::EMPTY];
        let mut nulls = [true];
        let mut values = [999];
        let result = reader.gather_column(
            &rows,
            0,
            &mut spans,
            &mut nulls,
            ColumnBuffer::I64(&mut values),
        );
        assert!(matches!(
            result,
            Err(SpoolError::Tag | SpoolError::Length | SpoolError::Identity)
        ));
        assert_eq!(values, [999]);
        assert_eq!(nulls, [true]);
    }
    for (kind, value) in [
        (DataType::Boolean, Value::Boolean(false)),
        (DataType::Text, Value::from("x")),
        (DataType::Timestamp, Value::Timestamp(DateTime::UNIX_EPOCH)),
    ] {
        let columns = specs(&[kind]);
        let (mut bytes, mut rows, summary) = fixture(&columns, &[(1, vec![value])]);
        match kind {
            DataType::Boolean => bytes[96] = 2,
            DataType::Text => bytes[96] = 255,
            DataType::Timestamp => bytes[104..108].copy_from_slice(&u32::MAX.to_le_bytes()),
            _ => unreachable!(),
        }
        let crc = crc32fast::hash(&bytes[96..]);
        bytes[80..84].copy_from_slice(&crc.to_le_bytes());
        resign_single_column(&mut bytes, &mut rows[0]);
        let source = Bytes::new(&bytes);
        let mut window = [0; 128];
        let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
        let mut spans = [CellSpan::EMPTY];
        let mut nulls = [false];
        let mut boolean = [false];
        let mut string = [0];
        let mut offsets = [(0, 0)];
        let mut stamp = [DateTime::UNIX_EPOCH];
        let output = match kind {
            DataType::Boolean => ColumnBuffer::Bool(&mut boolean),
            DataType::Text => ColumnBuffer::Variable {
                bytes: &mut string,
                offsets: &mut offsets,
            },
            _ => ColumnBuffer::Timestamps(&mut stamp),
        };
        assert!(matches!(
            reader.gather_column(&rows, 0, &mut spans, &mut nulls, output),
            Err(SpoolError::Tag | SpoolError::Utf8 | SpoolError::Timestamp)
        ));
    }
}

#[test]
fn run_record_corruption_incomplete_receipts_and_zero_column_rows_are_checked() {
    let columns = specs(&[DataType::Integer]);
    let (_, rows, summary) = fixture(
        &columns,
        &[(1, vec![Value::Integer(4)]), (2, vec![Value::Integer(5)])],
    );
    let mut sink = Cursor::new(Vec::new());
    let mut keys = [RowPosition::EMPTY; 2];
    let mut output = [0; 128];
    let mut runs = RowRunWriter::new(
        &mut sink,
        binding(&columns),
        SpoolLimits::default(),
        &mut keys,
        &mut output,
    )
    .unwrap();
    assert!(matches!(runs.push(rows[1]), Err(SpoolError::Identity))); // Must carry every contiguous append receipt.
    runs.push(rows[0]).unwrap();
    assert!(matches!(runs.finish(summary), Err(SpoolError::Identity)));
    runs.push(rows[1]).unwrap();
    let runs = runs.finish(summary).unwrap();
    sink.get_mut()[8] ^= 1;
    let source = Bytes::new(sink.get_ref());
    let mut input = [0; 128];
    let mut sorted = SortedRows::new(&source, runs, &mut input).unwrap();
    assert!(matches!(sorted.next_position(), Err(SpoolError::Checksum)));
    assert!(matches!(sorted.next_position(), Err(SpoolError::Poisoned)));
    let empty: [ColumnSpec; 0] = [];
    let (bytes, rows, summary) = fixture(&empty, &[(7, vec![])]);
    assert_eq!(bytes.len(), HEADER_BYTES);
    assert_eq!(summary.row_count(), 1);
    let mut sink = Cursor::new(Vec::new());
    let mut run = RowRunWriter::new(
        &mut sink,
        binding(&empty),
        SpoolLimits::default(),
        &mut keys,
        &mut output,
    )
    .unwrap();
    run.push(rows[0]).unwrap();
    assert!(run.finish(summary).unwrap().is_sorted());
}

#[test]
fn valid_short_writes_and_panicking_reads_keep_the_io_contract() {
    let columns = specs(&[DataType::Integer]);
    let mut sink = WriterFault {
        inner: Cursor::new(Vec::new()),
        remaining: 1000,
        lie: false,
        interrupt: true,
        panic: false,
    };
    let mut io = [0; 64];
    let mut writer = SpoolWriter::new(
        &mut sink,
        binding(&columns),
        SpoolLimits::default(),
        &mut io,
    )
    .unwrap();
    let row = writer.append(captured(1, &[Value::Integer(99)])).unwrap();
    let summary = writer.finish().unwrap();
    assert_eq!(sink.inner.get_ref().len(), summary.byte_len() as usize);
    struct Panics;
    impl ReadAt for Panics {
        fn read_at(&self, _: u64, _: &mut [u8]) -> io::Result<usize> {
            panic!("injected read panic")
        }
    }
    let mut window = [0; 64];
    let mut reader = SpoolReader::new(&Panics, summary, &mut window).unwrap();
    let mut spans = [CellSpan::EMPTY];
    let mut nulls = [false];
    let mut values = [0];
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reader
            .gather_column(
                &[row],
                0,
                &mut spans,
                &mut nulls,
                ColumnBuffer::I64(&mut values),
            )
            .map(|_| ())
    }))
    .is_err());
    assert!(matches!(
        reader.gather_column(
            &[row],
            0,
            &mut spans,
            &mut nulls,
            ColumnBuffer::I64(&mut values)
        ),
        Err(SpoolError::Poisoned)
    ));
}

#[test]
fn prepared_groups_amortize_metadata_and_keep_capacity_and_corruption_guarantees() {
    let columns = specs(&[DataType::Integer; 32]);
    let rows: Vec<_> = (0..8)
        .map(|i| (i, (0..32).map(|j| Value::Integer(i * 100 + j)).collect()))
        .collect();
    let (bytes, rows, summary) = fixture(&columns, &rows);
    let mut io_counts = [0; 2];
    for prepared in [false, true] {
        let source = Bytes::new(&bytes);
        let mut window = [0; 64];
        let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
        let mut spans = [CellSpan::EMPTY; 8];
        let mut nulls = [true; 8];
        let mut values = [999; 8];
        if prepared {
            let mut group = reader.prepare_group(&rows).unwrap();
            assert!(matches!(
                group.gather_column(
                    0,
                    &mut spans,
                    &mut nulls,
                    ColumnBuffer::I64(&mut values[..7])
                ),
                Err(SpoolError::BufferTooSmall)
            ));
            assert_eq!(values, [999; 8]);
            assert_eq!(nulls, [true; 8]);
            for column in 0..32 {
                group
                    .gather_column(
                        column,
                        &mut spans,
                        &mut nulls,
                        ColumnBuffer::I64(&mut values),
                    )
                    .unwrap();
                for (i, &value) in values.iter().enumerate() {
                    assert_eq!(value, i as i64 * 100 + i64::from(column));
                }
            }
        } else {
            for column in 0..32 {
                reader
                    .gather_column(
                        &rows,
                        column,
                        &mut spans,
                        &mut nulls,
                        ColumnBuffer::I64(&mut values),
                    )
                    .unwrap();
            }
        }
        io_counts[usize::from(prepared)] = source.calls.get();
    }
    assert!(
        io_counts[1] * 3 < io_counts[0],
        "one-shot/prepared read calls: {io_counts:?}"
    );
    let mut corrupt = bytes;
    *corrupt.last_mut().unwrap() ^= 1;
    let source = Bytes::new(&corrupt);
    let mut window = [0; 64];
    let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
    let mut group = reader.prepare_group(&rows).unwrap(); // All descriptor metadata remains valid.
    let mut spans = [CellSpan::EMPTY; 8];
    let mut nulls = [true; 8];
    let mut values = [999; 8];
    assert!(matches!(
        group.gather_column(31, &mut spans, &mut nulls, ColumnBuffer::I64(&mut values)),
        Err(SpoolError::Checksum)
    ));
    assert!(matches!(
        group.gather_column(0, &mut spans, &mut nulls, ColumnBuffer::I64(&mut values)),
        Err(SpoolError::Poisoned)
    ));
}

#[test]
fn buffered_append_receipts_do_not_certify_a_failed_finish() {
    let columns = specs(&[DataType::Integer]);
    for panic in [false, true] {
        let mut sink = WriterFault {
            inner: Cursor::new(Vec::new()),
            remaining: 12,
            lie: false,
            interrupt: false,
            panic,
        };
        let mut io = [0; 1024];
        let mut writer = SpoolWriter::new(
            &mut sink,
            binding(&columns),
            SpoolLimits::default(),
            &mut io,
        )
        .unwrap();
        let receipt = writer.append(captured(1, &[Value::Integer(99)])).unwrap();
        assert_eq!(receipt.row_id(), 1);
        assert!(writer.sink.inner.get_ref().is_empty()); // Receipt can precede physical bytes.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| writer.finish()));
        assert!(result.is_err() || result.unwrap().is_err());
        assert!(matches!(writer.finish(), Err(SpoolError::Poisoned)));
        assert!(matches!(
            writer.append(captured(2, &[Value::Integer(88)])),
            Err(SpoolError::Poisoned)
        ));
    }
}

#[test]
fn prepared_group_keeps_poison_after_a_later_io_panic() {
    struct Controlled<'a> {
        bytes: Bytes<'a>,
        panic: Cell<bool>,
    }
    impl ReadAt for Controlled<'_> {
        fn read_at(&self, offset: u64, out: &mut [u8]) -> io::Result<usize> {
            assert!(!self.panic.get(), "injected prepared read panic");
            self.bytes.read_at(offset, out)
        }
    }
    let columns = specs(&[DataType::Integer]);
    let (bytes, rows, summary) = fixture(
        &columns,
        &[(1, vec![Value::Integer(1)]), (2, vec![Value::Integer(2)])],
    );
    let source = Controlled {
        bytes: Bytes::new(&bytes),
        panic: Cell::new(false),
    };
    let mut window = [0; 64];
    let mut reader = SpoolReader::new(&source, summary, &mut window).unwrap();
    let mut group = reader.prepare_group(&rows).unwrap();
    source.panic.set(true);
    let mut spans = [CellSpan::EMPTY; 2];
    let mut nulls = [true; 2];
    let mut values = [999; 2];
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        group
            .gather_column(0, &mut spans, &mut nulls, ColumnBuffer::I64(&mut values))
            .map(|_| ())
    }))
    .is_err());
    source.panic.set(false);
    assert!(matches!(
        group.gather_column(0, &mut spans, &mut nulls, ColumnBuffer::I64(&mut values)),
        Err(SpoolError::Poisoned)
    ));
}
