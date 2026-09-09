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
use std::io::{self, Read, Seek, SeekFrom, Write};
use stoolap::storage::volume::legacy_lz4::{DecodeError, RawLz4Decoder, HISTORY_BYTES};

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

fn write_extension(output: &mut impl Write, mut extra: usize) {
    let repeated = [255u8; 4096];
    while extra >= 255 {
        let length = (extra / 255).min(repeated.len());
        output.write_all(&repeated[..length]).unwrap();
        extra -= length * 255;
    }
    output.write_all(&[extra as u8]).unwrap();
}

#[test]
fn decoded_block_larger_than_32_mib_uses_only_80_kib_caller_scratch() {
    // A 512-byte raw period repeated 65,537 times exceeds 32 MiB. This proves
    // the raw decoder's bound, not V4 column-layout conversion (the later
    // adapter must test actual encoded values/offsets). Source and output are
    // real files; fixture construction and verification are outside the
    // allocator window. No full input/output Vec exists at any point.
    const VALUE_BYTES: usize = 512;
    const ROWS: usize = 65_537;
    const DECODED: usize = VALUE_BYTES * ROWS;
    let mut source = tempfile::tempfile().unwrap();
    let mut target = tempfile::tempfile().unwrap();
    let mut pattern = [0u8; VALUE_BYTES];
    for (index, byte) in pattern.iter_mut().enumerate() {
        *byte = (index % 251) as u8;
    }
    // One literal period, a long overlapping match at offset 512, and the
    // required final five literals. This is a raw block, not a framed stream.
    source.write_all(&[0xff]).unwrap();
    write_extension(&mut source, VALUE_BYTES - 15);
    source.write_all(&pattern).unwrap();
    source
        .write_all(&(VALUE_BYTES as u16).to_le_bytes())
        .unwrap();
    write_extension(&mut source, DECODED - VALUE_BYTES - 5 - 19);
    source.write_all(&[0x50]).unwrap();
    source.write_all(&pattern[VALUE_BYTES - 5..]).unwrap();
    let stored = source.stream_position().unwrap();
    source.seek(SeekFrom::Start(0)).unwrap();

    struct BulkWriter<'a> {
        file: &'a mut std::fs::File,
        calls: usize,
        largest: usize,
    }
    impl Write for BulkWriter<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            self.largest = self.largest.max(bytes.len());
            self.file.write(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            panic!("decoder must not flush")
        }
    }
    let mut history = [0xa5; HISTORY_BYTES];
    let mut input_scratch = [0; 8192];
    let mut output_scratch = [0; 8192];
    assert_eq!(
        history.len() + input_scratch.len() + output_scratch.len(),
        80 * 1024
    );
    let mut writer = BulkWriter {
        file: &mut target,
        calls: 0,
        largest: 0,
    };
    let stop = start_counting();
    let mut decoder =
        RawLz4Decoder::new(&mut history, &mut input_scratch, &mut output_scratch).unwrap();
    let result = decoder
        .decode(&mut source, &mut writer, stored, DECODED as u64)
        .unwrap();
    drop(stop);
    assert_eq!(CALLS.with(Cell::get), 0);
    assert_eq!(result.stored_bytes, stored);
    assert_eq!(result.decoded_bytes, DECODED as u64);
    assert_eq!(writer.largest, 8192);
    // Chunked output, independent of the match's 512-byte period. No timing
    // threshold is claimed while other build/test workloads may be running.
    assert!(writer.calls < DECODED / 4096);
    assert_eq!(target.metadata().unwrap().len(), DECODED as u64);
    target.seek(SeekFrom::Start(0)).unwrap();
    let mut verify = [0; 8192];
    let mut verified = 0;
    loop {
        let read = target.read(&mut verify).unwrap();
        if read == 0 {
            break;
        }
        for (offset, &byte) in verify[..read].iter().enumerate() {
            assert_eq!(byte, pattern[(verified + offset) % VALUE_BYTES]);
        }
        verified += read;
    }
    assert_eq!(verified, DECODED);
}

#[test]
fn malformed_offset_and_sticky_abort_allocate_nothing() {
    let mut history = [0; HISTORY_BYTES];
    let mut input = [0; 8];
    let mut output = [0; 8];
    let mut decoder = RawLz4Decoder::new(&mut history, &mut input, &mut output).unwrap();
    let mut source = &[0x10, b'a', 0, 0, 0][..];
    let stop = start_counting();
    assert!(matches!(
        decoder.decode(&mut source, &mut io::sink(), 5, 5),
        Err(DecodeError::OffsetZero)
    ));
    assert!(matches!(
        decoder.decode(&mut source, &mut io::sink(), 5, 5),
        Err(DecodeError::Aborted)
    ));
    drop(stop);
    assert_eq!(CALLS.with(Cell::get), 0);
}
