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

#![cfg(not(feature = "mimalloc"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use stoolap::core::DataType;
use stoolap::storage::volume::legacy_column::{
    CellRef, ColumnSpec, Encoding, RowSlot, ValidatedColumn,
};
use stoolap::storage::volume::legacy_lz4::{RawLz4Decoder, HISTORY_BYTES};

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct CountingAllocator;
fn count() {
    if COUNTING.try_with(Cell::get).unwrap_or(false) {
        let _ = CALLS.try_with(|calls| calls.set(calls.get() + 1));
    }
}
// SAFETY: All pointers, layouts and ownership pass unchanged to System.
// The thread-local counters do not allocate or access allocation contents.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
struct StopCounting;
impl Drop for StopCounting {
    fn drop(&mut self) {
        COUNTING.with(|counting| counting.set(false));
    }
}
fn start_counting() -> StopCounting {
    CALLS.with(|calls| calls.set(0));
    COUNTING.with(|counting| counting.set(true));
    StopCounting
}

fn vector_bytes(row: usize, out: &mut [u8; 512]) {
    for (lane, bytes) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let bits = match lane {
            0 => 0x7fc0_0000 | row as u32, // Preserve distinct NaN payloads.
            1 => 0x8000_0000,
            2 => 0x7f80_0000,
            _ => (row as u32).wrapping_mul(7919) ^ (lane as u32 * 97),
        };
        bytes.copy_from_slice(&bits.to_le_bytes());
    }
}

#[test]
fn actual_65536_row_vector_wire_over_32_mib_decodes_and_gathers_without_allocating() {
    // This is the real serialize_column_block(Bytes) layout: one-byte nulls,
    // u64 offset count, u64 offset/length pairs, u64 blob length, raw LE vector
    // payloads. Unit tests compare this layout against the existing serializer
    // and decoder. Here we stream construction to real files, never a full
    // block Vec. Fixture construction and OS/file buffering are not measured.
    const ROWS: usize = 65_536;
    const VALUE_BYTES: usize = 512;
    const DECODED: usize = ROWS * 17 + 16 + ROWS * VALUE_BYTES;
    const { assert!(DECODED > 32 * 1024 * 1024) };
    let mut fixture = BufWriter::new(tempfile::tempfile().unwrap());
    for row in 0..ROWS {
        fixture
            .write_all(&[u8::from(row.is_multiple_of(97))])
            .unwrap();
    }
    fixture.write_all(&(ROWS as u64).to_le_bytes()).unwrap();
    for row in 0..ROWS {
        fixture
            .write_all(&((row * VALUE_BYTES) as u64).to_le_bytes())
            .unwrap();
        fixture
            .write_all(&(VALUE_BYTES as u64).to_le_bytes())
            .unwrap();
    }
    fixture
        .write_all(&((ROWS * VALUE_BYTES) as u64).to_le_bytes())
        .unwrap();
    let mut expected = [0; VALUE_BYTES];
    for row in 0..ROWS {
        vector_bytes(row, &mut expected);
        fixture.write_all(&expected).unwrap();
    }
    let mut fixture = fixture.into_inner().unwrap();
    assert_eq!(fixture.stream_position().unwrap(), DECODED as u64);
    fixture.seek(SeekFrom::Start(0)).unwrap();

    // A valid all-literal raw LZ4 block; this tests the actual decoded column
    // envelope without relying on a whole-output compressor for its fixture.
    let mut compressed = BufWriter::new(tempfile::tempfile().unwrap());
    compressed.write_all(&[0xf0]).unwrap();
    let mut extension = DECODED - 15;
    let full = [255; 4096];
    while extension >= 255 {
        let count = (extension / 255).min(full.len());
        compressed.write_all(&full[..count]).unwrap();
        extension -= count * 255;
    }
    compressed.write_all(&[extension as u8]).unwrap();
    let mut copy = [0; 8192];
    loop {
        let count = fixture.read(&mut copy).unwrap();
        if count == 0 {
            break;
        }
        compressed.write_all(&copy[..count]).unwrap();
    }
    let mut compressed = compressed.into_inner().unwrap();
    let stored_len = compressed.stream_position().unwrap();
    compressed.seek(SeekFrom::Start(0)).unwrap();
    let mut spool = tempfile::tempfile().unwrap();
    let mut history = [0; HISTORY_BYTES];
    let mut input = [0; 8192];
    let mut output = [0; 8192];
    let mut scratch = [0; 4096];
    let mut slots = [RowSlot::default(); 256];
    let mut payload = [0; 65536];
    let scratch_bytes = history.len()
        + input.len()
        + output.len()
        + scratch.len()
        + std::mem::size_of_val(&slots)
        + payload.len()
        + expected.len();
    assert!(scratch_bytes < 160 * 1024);
    assert!(scratch_bytes < 16 * 1024 * 1024);
    let meter = start_counting();
    let mut decoder = RawLz4Decoder::new(&mut history, &mut input, &mut output).unwrap();
    decoder
        .decode(&mut compressed, &mut spool, stored_len, DECODED as u64)
        .unwrap();
    let spec = ColumnSpec {
        encoding: Encoding::Bytes {
            data_type: DataType::Vector,
        },
        row_count: ROWS as u32,
        offset: 0,
        decoded_len: DECODED as u64,
    };
    let mut adapter = ValidatedColumn::new(&mut spool, spec, &mut scratch).unwrap();
    let mut start = 0;
    let mut subdivisions = 0;
    while start < ROWS as u32 {
        let plan = adapter
            .plan_range(
                start,
                (start + slots.len() as u32).min(ROWS as u32),
                payload.len(),
                &mut scratch,
            )
            .unwrap();
        assert!(plan.end > start);
        let view = adapter
            .read_range(start, plan.end, &mut slots, &mut payload, &mut scratch)
            .unwrap();
        for local in 0..view.len() {
            let row = start as usize + local;
            if row.is_multiple_of(97) {
                assert_eq!(view.get(local), Some(CellRef::Null(DataType::Vector)));
            } else {
                vector_bytes(row, &mut expected);
                assert_eq!(
                    view.get(local),
                    Some(CellRef::Bytes {
                        data_type: DataType::Vector,
                        bytes: &expected
                    })
                );
            }
        }
        start = plan.end;
        subdivisions += 1;
    }
    assert!(subdivisions > 500);
    let allocations = CALLS.with(Cell::get);
    drop(meter);
    assert_eq!(
        allocations, 0,
        "decode, validate, plan, gather and borrowed access must allocate nothing"
    );
}
