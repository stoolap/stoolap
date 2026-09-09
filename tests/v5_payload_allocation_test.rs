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
use std::io::{self, Cursor};
use std::num::NonZeroU64;

use stoolap::core::DataType;
use stoolap::storage::volume::v5::column_block::{ColumnInput, ColumnLimits};
use stoolap::storage::volume::v5::directory::{
    DirectoryKey, Layout as VolumeLayout, LeafEntry, RowBounds, Section, VolumeShape,
    GLOBAL_COLUMN, KEY_REQUIRED,
};
use stoolap::storage::volume::v5::directory_writer::{DirectoryScratch, ENCODING_BYTES};
use stoolap::storage::volume::v5::envelope::{
    Codec, FileIdentity, Header, PageDescriptor, ReadLimits,
};
use stoolap::storage::volume::v5::group_metadata::GroupRecord;
use stoolap::storage::volume::v5::metadata_runs::{
    RunWriter, SortedReader, IO_BYTES, RECORD_BYTES,
};
use stoolap::storage::volume::v5::page_io::ReadAt;
use stoolap::storage::volume::v5::payload_writer::{ColumnSpec, PayloadWriter};
use stoolap::storage::volume::v5::row_identity::RowSource;

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
}
struct Counting;
fn count() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            CALLS.with(|calls| calls.set(calls.get() + 1));
        }
    });
}
// Test-only shim forwards the exact allocation/deallocation to System.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, n: usize) -> *mut u8 {
        count();
        unsafe { System.realloc(ptr, layout, n) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOC: Counting = Counting;
struct Stop;
impl Drop for Stop {
    fn drop(&mut self) {
        TRACK.with(|track| track.set(false));
    }
}
struct Bytes<'a>(&'a [u8]);
impl ReadAt for Bytes<'_> {
    fn read_at(&self, offset: u64, out: &mut [u8]) -> io::Result<usize> {
        let bytes = self
            .0
            .get(usize::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?..)
            .ok_or(io::ErrorKind::UnexpectedEof)?;
        let n = bytes.len().min(out.len());
        out[..n].copy_from_slice(&bytes[..n]);
        Ok(n)
    }
}

#[test]
fn payload_groups_columns_spool_directory_and_finish_allocate_nothing() {
    let header = Header::new(FileIdentity::new(1, 2, 3).unwrap());
    let limits = ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: 1024 * 1024,
        page_decoded_bytes: 1024 * 1024,
    };
    let specs = [ColumnSpec {
        data_type: DataType::Integer,
        vector_dimensions: None,
    }];
    let dummy = LeafEntry {
        key: DirectoryKey {
            section: Section::RowIds as u16,
            flags: KEY_REQUIRED,
            column: GLOBAL_COLUMN,
            ordinal: 0,
        },
        page: PageDescriptor {
            offset: 64,
            stored_len: 1,
            decoded_len: 1,
            stored_checksum: 0,
            codec: Codec::Raw,
        },
    };
    for (groups, compress) in [
        (0, false),
        (1, false),
        (65, false),
        (0, true),
        (1, true),
        (65, true),
    ] {
        // All capacity belongs to the caller and exists before measured work.
        let mut file_bytes = vec![0; 131072];
        let mut spool_bytes = vec![0; RECORD_BYTES * 260];
        let mut entries = [dummy; 260];
        let mut run_buffer = [0; IO_BYTES];
        let mut input_buffer = [0; IO_BYTES];
        let mut encoding = [0; 128];
        let mut compressed = [0; 256];
        let mut table = lz4_flex::block::CompressTable::large();
        let mut directory = DirectoryScratch::new();
        let mut directory_encoding = [0; ENCODING_BYTES];
        let mut file = Cursor::new(file_bytes.as_mut_slice());
        let mut spool = Cursor::new(spool_bytes.as_mut_slice());
        let shape = VolumeShape {
            layout: VolumeLayout::RowId,
            row_count: groups,
            column_count: 1,
            group_count: groups,
            rows: (groups > 0).then_some(RowBounds {
                min: 0,
                max: groups as i64 - 1,
            }),
            window: None,
        };
        CALLS.with(|calls| calls.set(0));
        TRACK.with(|track| track.set(true));
        let stop = Stop;
        let mut runs = RunWriter::new(&mut spool, &mut entries, &mut run_buffer).unwrap();
        let mut writer = PayloadWriter::new(
            &mut file,
            header,
            shape,
            None,
            &specs,
            limits,
            ColumnLimits::default(),
            &mut runs,
        )
        .unwrap();
        for group in 0..groups {
            let record = GroupRecord {
                row_start: group,
                row_count: 1,
                column_count: 1,
                rows: RowBounds {
                    min: group as i64,
                    max: group as i64,
                },
            };
            writer
                .begin_group(
                    record,
                    &[group as i64],
                    &[RowSource::Dml(NonZeroU64::new(1).unwrap())],
                    &mut encoding,
                )
                .unwrap();
            assert!(writer
                .write_column(&[false], ColumnInput::F64(&[1.0]), &mut encoding)
                .is_err());
            if compress {
                writer
                    .write_column_compressed(
                        &[false],
                        ColumnInput::I64(&[group as i64]),
                        &mut encoding,
                        &mut compressed,
                        &mut table,
                    )
                    .unwrap();
            } else {
                writer
                    .write_column(&[false], ColumnInput::I64(&[group as i64]), &mut encoding)
                    .unwrap();
            }
        }
        let mut completed = writer.finish_payloads().unwrap();
        let runs = runs.finish().unwrap();
        let source = Bytes(spool.get_ref());
        let mut sorted = SortedReader::new(&source, runs, &mut input_buffer).unwrap();
        let finished = completed
            .finish(
                || sorted.next_entry(),
                &mut directory,
                &mut directory_encoding,
            )
            .unwrap();
        std::hint::black_box(finished);
        drop(stop);
        assert_eq!(
            CALLS.with(Cell::get),
            0,
            "groups={groups}, compress={compress}"
        );
    }
}
