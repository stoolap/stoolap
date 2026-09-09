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
use crate::common::SmartString;
use crate::storage::volume::column::ColumnData;
use crate::storage::volume::format::{deserialize_column_block, serialize_column_block};
use std::cell::Cell;
use std::io::Cursor;
use std::rc::Rc;
use std::sync::Arc;

fn spec(encoding: Encoding, rows: usize, bytes: &[u8]) -> ColumnSpec {
    ColumnSpec {
        encoding,
        row_count: rows as u32,
        offset: 0,
        decoded_len: bytes.len() as u64,
    }
}

fn assert_cell(cell: CellRef<'_>, expected: &ColumnData, row: usize) {
    if expected.is_null(row) {
        assert_eq!(cell, CellRef::Null(expected.data_type()));
        return;
    }
    match (cell, expected) {
        (CellRef::Int64(actual), ColumnData::Int64 { values, .. }) => {
            assert_eq!(actual, values[row])
        }
        (CellRef::TimestampNanos(actual), ColumnData::TimestampNanos { values, .. }) => {
            assert_eq!(actual, values[row])
        }
        (CellRef::Float64(actual), ColumnData::Float64 { values, .. }) => {
            assert_eq!(actual.to_bits(), values[row].to_bits())
        }
        (CellRef::Boolean(actual), ColumnData::Boolean { values, .. }) => {
            assert_eq!(actual, values[row])
        }
        (CellRef::DictionaryId(actual), ColumnData::Dictionary { ids, .. }) => {
            assert_eq!(actual, ids[row])
        }
        (
            CellRef::Bytes { data_type, bytes },
            ColumnData::Bytes {
                offsets,
                data,
                ext_type,
                ..
            },
        ) => {
            assert_eq!(data_type, *ext_type);
            let (offset, len) = offsets[row];
            assert_eq!(bytes, &data[offset as usize..(offset + len) as usize]);
        }
        _ => panic!("different column/cell types"),
    }
}

#[test]
fn all_wire_encodings_match_existing_decoder_in_bounded_ranges() {
    for count in [0, 1, 17, 257] {
        let nulls: Vec<_> = (0..count).map(|row| row % 7 == 2).collect();
        let signed = [i64::MIN, -1, 0, i64::MAX, 123_456_789];
        let bits = [
            0,
            1 << 63,
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            0x7ff8_0000_0000_0123,
            0x7ff0_0000_0000_0123,
        ];
        let dictionary: Arc<[SmartString]> = Arc::from([
            SmartString::from(""),
            SmartString::from("東京"),
            SmartString::from("café"),
        ]);
        let columns = [
            (
                COL_INT64,
                ColumnData::Int64 {
                    values: (0..count).map(|r| signed[r % signed.len()]).collect(),
                    nulls: nulls.clone(),
                },
            ),
            (
                COL_FLOAT64,
                ColumnData::Float64 {
                    values: (0..count)
                        .map(|r| f64::from_bits(bits[r % bits.len()]))
                        .collect(),
                    nulls: nulls.clone(),
                },
            ),
            (
                COL_TIMESTAMP,
                ColumnData::TimestampNanos {
                    values: (0..count).map(|r| signed[r % signed.len()]).collect(),
                    nulls: nulls.clone(),
                },
            ),
            (
                COL_BOOLEAN,
                ColumnData::Boolean {
                    values: (0..count).map(|r| r % 2 == 0).collect(),
                    nulls: nulls.clone(),
                },
            ),
            (
                COL_DICTIONARY,
                ColumnData::Dictionary {
                    ids: (0..count)
                        .map(|r| if nulls[r] { u32::MAX } else { (r % 3) as u32 })
                        .collect(),
                    dictionary: dictionary.clone(),
                    nulls: nulls.clone(),
                },
            ),
            (
                COL_BYTES,
                ColumnData::Bytes {
                    data: (0..count * 8).map(|i| (i % 251) as u8).collect(),
                    offsets: (0..count).map(|r| ((r * 8) as u64, 8)).collect(),
                    ext_type: DataType::Vector,
                    nulls: nulls.clone(),
                },
            ),
        ];
        for (tag, column) in columns {
            let bytes = serialize_column_block(&column, 0, count);
            let decoded = deserialize_column_block(
                &bytes,
                tag,
                count,
                Some(dictionary.clone()),
                DataType::Vector,
            )
            .unwrap();
            let extra = if tag == COL_DICTIONARY {
                3
            } else {
                DataType::Vector as u32
            };
            let encoding = Encoding::from_directory(tag, extra).unwrap();
            for scratch_size in [17, 47, 1024] {
                let mut cursor = Cursor::new(&bytes);
                let mut scratch = vec![0; scratch_size];
                let mut adapter =
                    ValidatedColumn::new(&mut cursor, spec(encoding, count, &bytes), &mut scratch)
                        .unwrap();
                let mut slots = [RowSlot::default(); 13];
                let mut output = [0xcc; 64];
                let mut start = 0;
                while start < count as u32 {
                    let plan = adapter
                        .plan_range(
                            start,
                            (start + slots.len() as u32).min(count as u32),
                            output.len(),
                            &mut scratch,
                        )
                        .unwrap();
                    let view = adapter
                        .read_range(plan.start, plan.end, &mut slots, &mut output, &mut scratch)
                        .unwrap();
                    assert_eq!(view.start(), start);
                    assert_eq!(view.payload_bytes() as u64, plan.payload_bytes);
                    for row in 0..view.len() {
                        assert_cell(view.get(row).unwrap(), &decoded, start as usize + row);
                    }
                    assert_eq!(view.get(view.len()), None);
                    start = plan.end;
                }
                assert!(adapter
                    .read_range(count as u32, count as u32, &mut [], &mut [], &mut scratch)
                    .unwrap()
                    .is_empty());
            }
        }
    }
}

fn bytes_wire(flags: &[u8], offsets: &[(u64, u64)], blob: &[u8]) -> Vec<u8> {
    let mut wire = flags.to_vec();
    wire.extend_from_slice(&(offsets.len() as u64).to_le_bytes());
    for (offset, len) in offsets {
        wire.extend_from_slice(&offset.to_le_bytes());
        wire.extend_from_slice(&len.to_le_bytes());
    }
    wire.extend_from_slice(&(blob.len() as u64).to_le_bytes());
    wire.extend_from_slice(blob);
    wire
}

#[test]
fn noncanonical_bytes_offsets_and_null_payloads_preserve_existing_semantics() {
    let bytes = bytes_wire(
        &[0, 0, 0, 1, 0],
        &[(5, 3), (1, 6), (8, 0), (0, 8), (0, 2)],
        b"abcdefgh",
    );
    let expected = deserialize_column_block(&bytes, COL_BYTES, 5, None, DataType::Json).unwrap();
    let mut cursor = Cursor::new(&bytes);
    let mut scratch = [0; 1024];
    let mut adapter = ValidatedColumn::new(
        &mut cursor,
        spec(
            Encoding::Bytes {
                data_type: DataType::Json,
            },
            5,
            &bytes,
        ),
        &mut scratch,
    )
    .unwrap();
    let mut rows = [RowSlot::default(); 5];
    let mut payload = [0; 11];
    assert_eq!(adapter.plan_range(0, 5, 9, &mut scratch).unwrap().end, 4);
    let view = adapter
        .read_range(0, 5, &mut rows, &mut payload, &mut scratch)
        .unwrap();
    for row in 0..5 {
        assert_cell(view.get(row).unwrap(), &expected, row);
    }
    assert_eq!(view.payload_bytes(), 11);
}

#[test]
fn malformed_layouts_flags_offsets_ids_and_truncations_fail_closed() {
    let fixed = [0u8; 18];
    let bytes = bytes_wire(&[0, 1], &[(0, 3), (2, 2)], b"abcd");
    for (encoding, wire) in [
        (Encoding::Int64, fixed.as_slice()),
        (
            Encoding::Bytes {
                data_type: DataType::Vector,
            },
            bytes.as_slice(),
        ),
    ] {
        for end in 0..wire.len() {
            let mut cursor = Cursor::new(&wire[..end]);
            assert!(
                ValidatedColumn::new(&mut cursor, spec(encoding, 2, wire), &mut [0; 17]).is_err()
            );
        }
        let mut cursor = Cursor::new(wire);
        let mut input = spec(encoding, 2, wire);
        input.decoded_len += 1;
        assert!(ValidatedColumn::new(&mut cursor, input, &mut [0; 17]).is_err());
    }
    let mut malformed = Vec::new();
    for position in [0, 1] {
        let mut wire = fixed.to_vec();
        wire[position] = 2;
        malformed.push((Encoding::Int64, wire));
    }
    malformed.push((Encoding::Boolean, vec![0, 1, 0, 2])); // NULL lane still validates.
    malformed.push((
        Encoding::Dictionary { entries: 1 },
        vec![0, 1, 1, 0, 0, 0, 255, 255, 255, 255],
    ));
    for (offset, value) in [
        (2, 3u64),
        (10, u64::MAX),
        (18, u64::MAX),
        (26, 4),
        (34, u64::MAX),
        (42, 3),
    ] {
        let mut wire = bytes.clone();
        wire[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        malformed.push((
            Encoding::Bytes {
                data_type: DataType::Vector,
            },
            wire,
        ));
    }
    for (encoding, wire) in malformed {
        let mut cursor = Cursor::new(&wire);
        assert!(ValidatedColumn::new(&mut cursor, spec(encoding, 2, &wire), &mut [0; 17]).is_err());
    }
    assert!(Encoding::from_directory(0, 0).is_err());
    assert!(Encoding::from_directory(COL_BYTES, 256).is_err());
    let mut cursor = Cursor::new([]);
    let input = ColumnSpec {
        encoding: Encoding::Int64,
        row_count: MAX_ROWS + 1,
        offset: 0,
        decoded_len: 0,
    };
    assert!(matches!(
        ValidatedColumn::new(&mut cursor, input, &mut [0; 17]),
        Err(ColumnError::InvalidRowCount)
    ));
    let input = ColumnSpec {
        row_count: 0,
        offset: u64::MAX,
        decoded_len: 1,
        ..input
    };
    assert!(matches!(
        ValidatedColumn::new(&mut cursor, input, &mut [0; 17]),
        Err(ColumnError::LengthOverflow)
    ));
}

#[test]
fn buffer_preflight_does_not_mutate_output_or_poison_handle() {
    let bytes = bytes_wire(&[0, 1, 0], &[(0, 8), (0, 8), (8, 2)], b"abcdefghij");
    let mut cursor = Cursor::new(&bytes);
    let mut scratch = [0; 53];
    let mut adapter = ValidatedColumn::new(
        &mut cursor,
        spec(
            Encoding::Bytes {
                data_type: DataType::Json,
            },
            3,
            &bytes,
        ),
        &mut scratch,
    )
    .unwrap();
    let sentinel = RowSlot {
        offset: None,
        len: 4,
    };
    let mut rows = [sentinel; 3];
    let mut output = [0xcc; 9];
    assert!(matches!(
        adapter.read_range(0, 3, &mut rows, &mut output, &mut scratch),
        Err(ColumnError::BufferTooSmall {
            rows: 3,
            payload_bytes: 10
        })
    ));
    assert_eq!(rows, [sentinel; 3]);
    assert_eq!(output, [0xcc; 9]);
    assert!(!adapter.is_aborted());
    assert!(matches!(
        adapter.plan_range(0, 3, 7, &mut scratch),
        Err(ColumnError::BufferTooSmall {
            rows: 1,
            payload_bytes: 8
        })
    ));
    assert!(!adapter.is_aborted());
    assert!(matches!(
        adapter.read_range(0, 3, &mut rows[..2], &mut output, &mut scratch),
        Err(ColumnError::BufferTooSmall { .. })
    ));
    assert_eq!(rows, [sentinel; 3]);
    assert_eq!(
        adapter.plan_range(1, 3, 0, &mut scratch).unwrap(),
        RangePlan {
            start: 1,
            end: 2,
            payload_bytes: 0
        }
    );
    let view = adapter
        .read_range(1, 3, &mut rows, &mut output, &mut scratch)
        .unwrap();
    assert_eq!(view.get(0), Some(CellRef::Null(DataType::Json)));
    assert_eq!(
        view.get(1),
        Some(CellRef::Bytes {
            data_type: DataType::Json,
            bytes: b"ij"
        })
    );
    assert_eq!(&output[2..], &[0xcc; 7]);
}

struct ControlledSpool {
    cursor: Cursor<Vec<u8>>,
    mode: Rc<Cell<u8>>,
    calls: Rc<Cell<usize>>,
    stop_at: u64,
    interrupt: bool,
}
impl Read for ControlledSpool {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.calls.set(self.calls.get() + 1);
        if self.mode.get() == 5 {
            self.interrupt = !self.interrupt;
            if self.interrupt {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let len = out.len().min(3);
            return self.cursor.read(&mut out[..len]);
        }
        if self.cursor.position() >= self.stop_at {
            match self.mode.get() {
                1 => return Err(io::ErrorKind::Other.into()),
                2 => panic!("injected spool panic"),
                3 => return Ok(out.len() + 1),
                _ => {}
            }
        }
        let len = if self.mode.get() == 0 {
            out.len()
        } else {
            out.len()
                .min(self.stop_at.saturating_sub(self.cursor.position()) as usize)
        };
        self.cursor.read(&mut out[..len])
    }
}
impl Seek for ControlledSpool {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.calls.set(self.calls.get() + 1);
        if self.mode.get() == 4 && matches!(position, SeekFrom::Start(_)) {
            return Ok(u64::MAX);
        }
        self.cursor.seek(position)
    }
}

#[test]
fn short_interrupted_io_nonzero_region_and_invalid_counts() {
    let bytes = bytes_wire(&[0], &[(0, 4)], b"data");
    for mode_value in [4, 5] {
        let mut backing = b"prefix!".to_vec();
        backing.extend_from_slice(&bytes);
        backing.extend_from_slice(b"suffix");
        let mode = Rc::new(Cell::new(mode_value));
        let mut spool = ControlledSpool {
            cursor: Cursor::new(backing),
            mode,
            calls: Rc::new(Cell::new(0)),
            stop_at: u64::MAX,
            interrupt: false,
        };
        let mut input = spec(
            Encoding::Bytes {
                data_type: DataType::Vector,
            },
            1,
            &bytes,
        );
        input.offset = 7;
        let mut scratch = [0; 17];
        let result = ValidatedColumn::new(&mut spool, input, &mut scratch);
        if mode_value == 4 {
            assert!(matches!(result, Err(ColumnError::InvalidSeekPosition)));
            continue;
        }
        let mut adapter = result.unwrap();
        let mut rows = [RowSlot::default(); 1];
        let mut output = [0; 4];
        assert_eq!(
            adapter
                .read_range(0, 1, &mut rows, &mut output, &mut scratch)
                .unwrap()
                .get(0),
            Some(CellRef::Bytes {
                data_type: DataType::Vector,
                bytes: b"data"
            })
        );
    }
    let mode = Rc::new(Cell::new(3));
    let mut spool = ControlledSpool {
        cursor: Cursor::new(vec![0; 9]),
        mode,
        calls: Rc::new(Cell::new(0)),
        stop_at: 0,
        interrupt: false,
    };
    assert!(matches!(
        ValidatedColumn::new(&mut spool, spec(Encoding::Int64, 1, &[0; 9]), &mut [0; 17]),
        Err(ColumnError::InvalidReadCount)
    ));
}

#[test]
fn partial_gather_io_error_and_panic_abort_before_callbacks_and_stay_aborted() {
    let bytes = bytes_wire(&[0, 0], &[(0, 4), (4, 4)], b"abcdefgh");
    for mode_value in [1, 2, 3] {
        let mode = Rc::new(Cell::new(0));
        let calls = Rc::new(Cell::new(0));
        let mut spool = ControlledSpool {
            cursor: Cursor::new(bytes.clone()),
            mode: mode.clone(),
            calls: calls.clone(),
            stop_at: bytes.len() as u64 - 3,
            interrupt: false,
        };
        let mut scratch = [0; 64];
        let mut adapter = ValidatedColumn::new(
            &mut spool,
            spec(
                Encoding::Bytes {
                    data_type: DataType::Json,
                },
                2,
                &bytes,
            ),
            &mut scratch,
        )
        .unwrap();
        mode.set(mode_value);
        let mut rows = [RowSlot::default(); 2];
        let mut output = [0; 8];
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            adapter
                .read_range(0, 2, &mut rows, &mut output, &mut scratch)
                .map(|_| ())
        }));
        match mode_value {
            1 => assert!(matches!(result, Ok(Err(ColumnError::Io(_))))),
            2 => assert!(result.is_err()),
            3 => assert!(matches!(result, Ok(Err(ColumnError::InvalidReadCount)))),
            _ => unreachable!(),
        }
        assert_eq!(&output[..5], b"abcde");
        assert!(adapter.is_aborted());
        let old_calls = calls.get();
        assert!(matches!(
            adapter.plan_range(0, 1, 8, &mut scratch),
            Err(ColumnError::Aborted)
        ));
        assert!(matches!(
            adapter.read_range(0, 1, &mut rows, &mut output, &mut scratch),
            Err(ColumnError::Aborted)
        ));
        assert_eq!(calls.get(), old_calls);
    }
}

#[test]
fn compact_slots_distinguish_null_and_nonnull_empty_at_zero() {
    assert_eq!(
        std::mem::size_of::<RowSlot>(),
        2 * std::mem::size_of::<usize>()
    );
    let bytes = bytes_wire(&[0, 1, 0], &[(0, 0), (0, 0), (0, 0)], b"");
    let mut cursor = Cursor::new(&bytes);
    let mut scratch = [0; 17];
    let mut adapter = ValidatedColumn::new(
        &mut cursor,
        spec(
            Encoding::Bytes {
                data_type: DataType::Json,
            },
            3,
            &bytes,
        ),
        &mut scratch,
    )
    .unwrap();
    let mut rows = [RowSlot::default(); 3];
    let view = adapter
        .read_range(0, 3, &mut rows, &mut [], &mut scratch)
        .unwrap();
    assert_eq!(
        view.get(0),
        Some(CellRef::Bytes {
            data_type: DataType::Json,
            bytes: b""
        })
    );
    assert_eq!(view.get(1), Some(CellRef::Null(DataType::Json)));
    assert_eq!(view.get(2), view.get(0));
}

#[test]
fn large_logical_lengths_remain_exact_without_address_space_allocation() {
    struct SparseSpool {
        prefix: Vec<u8>,
        position: u64,
        length: u64,
    }
    impl Read for SparseSpool {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            // This test's preflight must never read the large value itself.
            assert!(self.position < self.prefix.len() as u64);
            let start = self.position as usize;
            let count = out.len().min(self.prefix.len() - start);
            out[..count].copy_from_slice(&self.prefix[start..start + count]);
            self.position += count as u64;
            Ok(count)
        }
    }
    impl Seek for SparseSpool {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.position = match position {
                SeekFrom::Start(offset) => offset,
                SeekFrom::End(0) => self.length,
                _ => panic!("unexpected relative seek"),
            };
            Ok(self.position)
        }
    }
    let value_len = u32::MAX as u64 + 9;
    let mut prefix = bytes_wire(&[0], &[(0, value_len)], b"");
    prefix[25..33].copy_from_slice(&value_len.to_le_bytes());
    let mut spool = SparseSpool {
        length: prefix.len() as u64 + value_len,
        prefix,
        position: 0,
    };
    let input = ColumnSpec {
        encoding: Encoding::Bytes {
            data_type: DataType::Vector,
        },
        row_count: 1,
        offset: 0,
        decoded_len: spool.length,
    };
    let mut scratch = [0; 17];
    let mut adapter = ValidatedColumn::new(&mut spool, input, &mut scratch).unwrap();
    let sentinel = RowSlot {
        offset: None,
        len: 19,
    };
    let mut rows = [sentinel];
    let mut output = [0xa5; 64];
    assert!(
        matches!(adapter.read_range(0, 1, &mut rows, &mut output, &mut scratch), Err(ColumnError::BufferTooSmall { rows: 1, payload_bytes }) if payload_bytes == value_len)
    );
    assert_eq!(rows, [sentinel]);
    assert_eq!(output, [0xa5; 64]);
    assert!(!adapter.is_aborted());
}

#[test]
fn columns_with_different_byte_caps_gather_the_same_physical_range() {
    let bytes = bytes_wire(
        &[0, 0, 1, 0],
        &[(0, 4), (4, 1), (5, 3), (8, 2)],
        b"abcdefghij",
    );
    let integers = ColumnData::Int64 {
        values: vec![10, 11, 12, 13],
        nulls: vec![false; 4],
    };
    let fixed = serialize_column_block(&integers, 0, 4);
    let mut byte_spool = Cursor::new(&bytes);
    let mut fixed_spool = Cursor::new(&fixed);
    let mut scratch = [0; 64];
    let mut variable = ValidatedColumn::new(
        &mut byte_spool,
        spec(
            Encoding::Bytes {
                data_type: DataType::Json,
            },
            4,
            &bytes,
        ),
        &mut scratch,
    )
    .unwrap();
    let mut integer = ValidatedColumn::new(
        &mut fixed_spool,
        spec(Encoding::Int64, 4, &fixed),
        &mut scratch,
    )
    .unwrap();
    let mut variable_slots = [RowSlot::default(); 4];
    let mut integer_slots = [RowSlot::default(); 4];
    let mut variable_payload = [0; 4];
    let mut integer_payload = [0; 16];
    let mut start = 0;
    while start < 4 {
        let byte_plan = variable
            .plan_range(start, 4, variable_payload.len(), &mut scratch)
            .unwrap();
        let integer_plan = integer
            .plan_range(start, 4, integer_payload.len(), &mut scratch)
            .unwrap();
        let end = byte_plan.end.min(integer_plan.end);
        assert!(end > start);
        let byte_view = variable
            .read_range(
                start,
                end,
                &mut variable_slots,
                &mut variable_payload,
                &mut scratch,
            )
            .unwrap();
        let integer_view = integer
            .read_range(
                start,
                end,
                &mut integer_slots,
                &mut integer_payload,
                &mut scratch,
            )
            .unwrap();
        assert_eq!(byte_view.len(), integer_view.len());
        for local in 0..integer_view.len() {
            assert_eq!(
                integer_view.get(local),
                Some(CellRef::Int64(10 + start as i64 + local as i64))
            );
        }
        start = end;
    }
}
