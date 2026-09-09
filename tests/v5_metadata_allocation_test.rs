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
use std::io::{self, Cursor, Seek, SeekFrom, Write};

use stoolap::storage::volume::v5::directory::{DirectoryKey, LeafEntry, Section, KEY_REQUIRED};
use stoolap::storage::volume::v5::envelope::{Codec, PageDescriptor};
use stoolap::storage::volume::v5::metadata_runs::{
    merge_pass, MergeScratch, RunWriter, SortedReader, IO_BYTES, RECORD_BYTES,
};
use stoolap::storage::volume::v5::page_io::ReadAt;

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct Counter;
fn record() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            CALLS.with(|calls| calls.set(calls.get() + 1));
        }
    });
}
// Test-only shim; allocation/deallocation is delegated unchanged to System.
unsafe impl GlobalAlloc for Counter {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counter = Counter;
struct Stop;
impl Drop for Stop {
    fn drop(&mut self) {
        TRACK.with(|track| track.set(false));
    }
}

// Preallocated memory only simulates the two scratch files for this unit.
// Their test backing is outside the meter; production supplies disk streams.
struct Spool(Cursor<Vec<u8>>);
impl Write for Spool {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
impl Seek for Spool {
    fn seek(&mut self, offset: SeekFrom) -> io::Result<u64> {
        self.0.seek(offset)
    }
}
impl ReadAt for Spool {
    fn read_at(&self, offset: u64, bytes: &mut [u8]) -> io::Result<usize> {
        let offset = usize::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?;
        let source = self.0.get_ref().get(offset..).unwrap_or_default();
        let count = bytes.len().min(source.len());
        bytes[..count].copy_from_slice(&source[..count]);
        Ok(count)
    }
}
fn entry(n: u64) -> LeafEntry {
    LeafEntry {
        key: DirectoryKey {
            section: Section::ColumnBlocks as u16,
            flags: KEY_REQUIRED,
            column: 0,
            ordinal: n,
        },
        page: PageDescriptor {
            offset: 64 + n * 8,
            stored_len: 8,
            decoded_len: 8,
            stored_checksum: n as u32,
            codec: Codec::Raw,
        },
    }
}

#[test]
fn sorting_spilling_merging_and_complete_read_use_only_caller_buffers() {
    for count in [0, 1, 17, 18, 65, 10001] {
        let mut a = Spool(Cursor::new(Vec::with_capacity(count * RECORD_BYTES)));
        let mut b = Spool(Cursor::new(Vec::with_capacity(count * RECORD_BYTES)));
        let mut entries = [entry(0); 17];
        let mut output = [0; IO_BYTES];
        let mut scratch = MergeScratch::new();
        assert_eq!(std::mem::size_of::<MergeScratch>(), 3 * IO_BYTES);
        CALLS.with(|calls| calls.set(0));
        TRACK.with(|track| track.set(true));
        {
            let _stop = Stop;
            let mut writer = RunWriter::new(&mut a, &mut entries, &mut output).unwrap();
            for n in (0..count as u64).rev() {
                writer.push(entry(n)).unwrap();
            }
            let mut runs = writer.finish().unwrap();
            while !runs.is_sorted() {
                runs = merge_pass(&a, &mut b, runs, &mut scratch).unwrap();
                std::mem::swap(&mut a, &mut b);
            }
            let mut reader = SortedReader::new(&a, runs, &mut output).unwrap();
            for n in 0..count as u64 {
                assert_eq!(reader.next_entry().unwrap(), Some(entry(n)));
            }
            assert_eq!(reader.next_entry().unwrap(), None);
            assert!(reader.is_complete());
        }
        assert_eq!(CALLS.with(Cell::get), 0, "record count {count}");
    }
}
