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

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use chrono::{DateTime, Utc};

use stoolap::common::SmartString;
use stoolap::core::DataType;
use stoolap::storage::volume::v5::column_block::{
    ColumnBlockRef, ColumnEncodePlan, ColumnError, ColumnExpectation, ColumnIdentity, ColumnInput,
    ColumnLimits, MAX_DECODED_BYTES,
};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct CountAlloc;
fn count() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            CALLS.with(|calls| calls.set(calls.get() + 1));
        }
    });
}
// Test-only instrumentation delegates every allocation unchanged to System.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(p, layout, size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        unsafe { System.dealloc(p, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;
struct Stop;
impl Drop for Stop {
    fn drop(&mut self) {
        TRACK.with(|track| track.set(false));
    }
}
fn expected(dt: DataType, rows: usize) -> ColumnExpectation {
    ColumnExpectation {
        identity: ColumnIdentity {
            physical_column: 0,
            group: 7,
            row_start: 1_000_000,
            row_count: rows as u32,
            data_type: dt,
        },
        vector_dimensions: None,
    }
}

#[test]
fn column_codecs_allocate_nothing_for_all_encodings_and_rejections() {
    for rows in [10, 4096] {
        let nulls: Vec<bool> = (0..rows).map(|i| i % 7 == 0).collect();
        let all_null = vec![true; rows];
        let ints: Vec<i64> = (0..rows as i64).collect();
        let floats: Vec<f64> = ints.iter().map(|&i| i as f64).collect();
        let booleans: Vec<bool> = ints.iter().map(|&i| i % 2 == 0).collect();
        let timestamps: Vec<DateTime<Utc>> = ints
            .iter()
            .map(|&i| DateTime::from_timestamp_nanos(i))
            .collect();
        let wide_timestamps: Vec<DateTime<Utc>> = (0..rows)
            .map(|i| match i % 3 {
                0 => DateTime::<Utc>::MIN_UTC,
                1 => DateTime::<Utc>::MAX_UTC,
                _ => DateTime::from_timestamp(1_483_228_799, 1_500_000_000).unwrap(),
            })
            .collect();
        let ids: Vec<u32> = (0..rows).map(|i| (i % 2) as u32).collect();
        let dictionary = [SmartString::from("a"), SmartString::from("z")];
        let spans = vec![(0, 4); rows];
        let inputs = [
            (DataType::Integer, ColumnInput::I64(&ints)),
            (DataType::Float, ColumnInput::F64(&floats)),
            (DataType::Timestamp, ColumnInput::TimestampNanos(&ints)),
            (DataType::Timestamp, ColumnInput::Timestamps(&timestamps)),
            (
                DataType::Timestamp,
                ColumnInput::Timestamps(&wide_timestamps),
            ),
            (DataType::Boolean, ColumnInput::Bool(&booleans)),
            (
                DataType::Text,
                ColumnInput::Variable {
                    data: b"text",
                    offsets: &spans,
                },
            ),
            (
                DataType::Json,
                ColumnInput::Variable {
                    data: b"nope",
                    offsets: &spans,
                },
            ),
            (
                DataType::Vector,
                ColumnInput::Variable {
                    data: &[0, 0, 0x80, 0x7f],
                    offsets: &spans,
                },
            ),
            (
                DataType::Text,
                ColumnInput::DictionaryText {
                    ids: &ids,
                    dictionary: &dictionary,
                },
            ),
            (
                DataType::Text,
                ColumnInput::PlainTextFromDictionary {
                    ids: &ids,
                    dictionary: &dictionary,
                },
            ),
        ];
        let mut output = vec![0; MAX_DECODED_BYTES];
        CALLS.with(|calls| calls.set(0));
        TRACK.with(|track| track.set(true));
        {
            let _stop = Stop;
            for _ in 0..8 {
                for &(dt, input) in &inputs {
                    let e = expected(dt, rows);
                    let plan =
                        ColumnEncodePlan::new(e, &nulls, input, ColumnLimits::default()).unwrap();
                    let len = plan.encode_into(&mut output).unwrap();
                    let block =
                        ColumnBlockRef::parse(&output[..len], e, ColumnLimits::default()).unwrap();
                    for cell in block.cells() {
                        black_box(cell);
                    }
                    black_box(block.fixed_values());
                    if matches!(input, ColumnInput::DictionaryText { .. }) {
                        assert_eq!(block.dictionary_lookup("z").unwrap(), Some(1));
                        assert_eq!(block.dictionary_lookup("missing").unwrap(), None);
                    }
                    if output[9] == 4 {
                        // Invalid Chrono pair with valid page shape rejects
                        // without allocation, after the first (NULL) row.
                        let subsec = 64 + rows.div_ceil(8) + 12 + 8;
                        output[subsec..subsec + 4].copy_from_slice(&2_000_000_000u32.to_le_bytes());
                        assert!(matches!(
                            ColumnBlockRef::parse(&output[..len], e, ColumnLimits::default()),
                            Err(ColumnError::Timestamp)
                        ));
                        plan.encode_into(&mut output).unwrap();
                    }
                    // Valid envelope length but invalid typed reserved field.
                    output[56] = 1;
                    assert!(matches!(
                        ColumnBlockRef::parse(&output[..len], e, ColumnLimits::default()),
                        Err(ColumnError::Reserved)
                    ));
                    assert!(matches!(
                        plan.encode_into(&mut output[..len - 1]),
                        Err(ColumnError::OutputTooShort)
                    ));
                }
                let e = expected(DataType::Json, rows);
                let plan = ColumnEncodePlan::new(
                    e,
                    &all_null,
                    ColumnInput::AllNull,
                    ColumnLimits::default(),
                )
                .unwrap();
                let len = plan.encode_into(&mut output).unwrap();
                black_box(
                    ColumnBlockRef::parse(&output[..len], e, ColumnLimits::default()).unwrap(),
                );
            }
        }
        assert_eq!(CALLS.with(Cell::get), 0, "row count {rows}");
    }
}

#[test]
fn selected_plain_text_rows_do_not_rebuild_a_large_unsorted_source_dictionary() {
    let dictionary: Vec<SmartString> = (0..10_000)
        .rev()
        .map(|i| SmartString::from(format!("value-{i}")))
        .collect();
    let ids = [9_999, 2, 0];
    let e = expected(DataType::Text, ids.len());
    let mut output = [0; 256];
    CALLS.with(|calls| calls.set(0));
    TRACK.with(|track| track.set(true));
    {
        let _stop = Stop;
        for _ in 0..128 {
            let plan = ColumnEncodePlan::new(
                e,
                &[false; 3],
                ColumnInput::PlainTextFromDictionary {
                    ids: &ids,
                    dictionary: &dictionary,
                },
                ColumnLimits::default(),
            )
            .unwrap();
            let len = plan.encode_into(&mut output).unwrap();
            let block = ColumnBlockRef::parse(&output[..len], e, ColumnLimits::default()).unwrap();
            for (i, &id) in ids.iter().enumerate() {
                let stoolap::storage::volume::v5::column_block::ColumnCell::Text(text) =
                    block.cell(i).unwrap()
                else {
                    panic!("text cell expected")
                };
                assert_eq!(text, dictionary[id as usize].as_str());
            }
        }
    }
    assert_eq!(CALLS.with(Cell::get), 0);
}

#[test]
fn reused_compression_tables_never_allocate_or_upgrade_in_the_page_loop() {
    use lz4_flex::block::CompressTable;
    use stoolap::storage::volume::v5::compression::{CompressionError, CompressionPlan};
    use stoolap::storage::volume::v5::envelope::{Codec, ReadLimits};

    let limits = ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: MAX_DECODED_BYTES as u64,
        page_decoded_bytes: MAX_DECODED_BYTES as u64,
    };
    let input: Vec<u8> = (0..MAX_DECODED_BYTES).map(|i| (i % 13) as u8).collect();
    let mut output = vec![
        0;
        CompressionPlan::new(input.len(), &limits)
            .unwrap()
            .output_capacity()
    ];
    let mut decoded = vec![0; input.len()];
    let mut small = CompressTable::small();
    let mut large = CompressTable::large();
    // Backing buffers and table construction are outside the operation meter.
    CALLS.with(|calls| calls.set(0));
    TRACK.with(|track| track.set(true));
    {
        let _stop = Stop;
        for _ in 0..4 {
            for length in [1, 4096, 65_534, 65_535, MAX_DECODED_BYTES, 17] {
                let plan = CompressionPlan::new(length, &limits).unwrap();
                if plan.requires_large_table() {
                    assert!(matches!(
                        plan.compress(&input[..length], &mut output, &mut small),
                        Err(CompressionError::LargeTableRequired)
                    ));
                    assert!(matches!(small, CompressTable::Small(_)));
                } else {
                    black_box(
                        plan.compress(&input[..length], &mut output, &mut small)
                            .unwrap(),
                    );
                }
                let block = plan
                    .compress(&input[..length], &mut output, &mut large)
                    .unwrap();
                match block.codec() {
                    Codec::Raw => assert_eq!(block.bytes().as_ptr(), input.as_ptr()),
                    Codec::Lz4Block => {
                        assert_eq!(
                            lz4_flex::block::decompress_into(block.bytes(), &mut decoded[..length])
                                .unwrap(),
                            length
                        );
                        assert_eq!(decoded[..length], input[..length]);
                    }
                }
            }
        }
    }
    assert_eq!(CALLS.with(Cell::get), 0);
    // Positive control: the dependency's convenience API allocates its table.
    CALLS.with(|calls| calls.set(0));
    TRACK.with(|track| track.set(true));
    {
        let _stop = Stop;
        black_box(lz4_flex::block::compress_into(&input, &mut output).unwrap());
    }
    assert!(CALLS.with(Cell::get) > 0);
}
