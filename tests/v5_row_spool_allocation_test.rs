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
use std::io::{self, Seek, SeekFrom, Write};
use std::num::NonZeroU64;

use std::fs::File;
use stoolap::core::{DataType, Value};
use stoolap::storage::volume::v5::column_block::{
    ColumnBlockRef, ColumnCell, ColumnInput, ColumnLimits,
};
use stoolap::storage::volume::v5::directory::{
    DirectoryKey, Layout as VolumeLayout, LeafEntry, RowBounds, Section, VolumeShape,
    GLOBAL_COLUMN, KEY_REQUIRED,
};
use stoolap::storage::volume::v5::directory::{RootSummary, MAX_NODE_BYTES};
use stoolap::storage::volume::v5::directory_reader::{DirectoryLookup, KeyIdentity};
use stoolap::storage::volume::v5::directory_writer::{DirectoryScratch, ENCODING_BYTES};
use stoolap::storage::volume::v5::envelope::{
    Codec, FileIdentity, Header, PageDescriptor, ReadLimits,
};
use stoolap::storage::volume::v5::group_metadata::GroupPageExpectation;
use stoolap::storage::volume::v5::group_metadata::GroupRecord;
use stoolap::storage::volume::v5::metadata_runs::{RunWriter, SortedReader, IO_BYTES};
use stoolap::storage::volume::v5::page_io::{OpenedEnvelope, PageReadPlan, ReadAt};
use stoolap::storage::volume::v5::payload_writer::{ColumnSpec, PayloadWriter};
use stoolap::storage::volume::v5::row_identity::RowSource;
use stoolap::storage::volume::v5::row_identity::SourceEncodingContext;
use stoolap::storage::volume::v5::row_identity::{IdentityPageExpectation, SourceContext};
use stoolap::storage::volume::v5::row_spool::runs::{merge_pass, RowRunWriter, SortedRows};
use stoolap::storage::volume::v5::row_spool::{
    CapturedRow, CellSpan, ColumnBuffer, RowPosition, SpoolBinding, SpoolLimits, SpoolReader,
    SpoolWriter,
};

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
struct CountedFile {
    file: File,
    writes: usize,
    written: u64,
}
impl std::ops::Deref for CountedFile {
    type Target = File;
    fn deref(&self) -> &File {
        &self.file
    }
}
impl Write for CountedFile {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        let n = self.file.write(bytes)?;
        self.written += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}
impl Seek for CountedFile {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.file.seek(from)
    }
}
struct FileSource<'a> {
    file: &'a File,
    calls: Cell<usize>,
    bytes: Cell<usize>,
}
impl<'a> FileSource<'a> {
    fn new(file: &'a File) -> Self {
        Self {
            file,
            calls: Cell::new(0),
            bytes: Cell::new(0),
        }
    }
}
impl ReadAt for FileSource<'_> {
    fn read_at(&self, offset: u64, output: &mut [u8]) -> io::Result<usize> {
        self.calls.set(self.calls.get() + 1);
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(self.file, output, offset)?;
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(self.file, output, offset)?;
        self.bytes.set(self.bytes.get() + n);
        Ok(n)
    }
}

#[test]
fn captured_rows_external_sort_and_payload_writer_use_only_reserved_scratch() {
    const ROWS: usize = 1025;
    const GROUP: usize = 64;
    // File ownership/opening and all caller buffers are outside the allocation
    // meter. The measured pipeline never materializes all captured rows/values.
    let mut payload = CountedFile {
        file: tempfile::tempfile().unwrap(),
        writes: 0,
        written: 0,
    };
    let mut first = tempfile::tempfile().unwrap();
    let mut second = tempfile::tempfile().unwrap();
    let mut volume = tempfile::tempfile().unwrap();
    let mut descriptors = tempfile::tempfile().unwrap();
    let header = Header::new(FileIdentity::new(2, 3, 4).unwrap());
    let columns = [
        ColumnSpec {
            data_type: DataType::Integer,
            vector_dimensions: None,
        },
        ColumnSpec {
            data_type: DataType::Text,
            vector_dimensions: None,
        },
    ];
    let binding = SpoolBinding {
        identity: header.identity,
        schema_version: NonZeroU64::new(9).unwrap(),
        columns: &columns,
        source: SourceEncodingContext::for_staged_file(&header, None).unwrap(),
    };
    let limits = SpoolLimits::default();
    let mut values = [Value::Integer(0), Value::from("captured λ payload")];
    let mut row_keys = [RowPosition::EMPTY; 17];
    let mut write_buffer = [0; 4096];
    let mut run_output = [0; 4096];
    let (mut left, mut right, mut merge_output) = ([0; 4096], [0; 4096], [0; 4096]);
    let mut positions = [RowPosition::EMPTY; GROUP];
    let mut row_ids = [0; GROUP];
    let mut row_sources = [RowSource::Dml(NonZeroU64::MIN); GROUP];
    let mut read_window = [0; 8192];
    let mut sorted_input = [0; 4096];
    let mut spans = [CellSpan::EMPTY; GROUP];
    let mut nulls = [false; GROUP];
    let mut integers = [0; GROUP];
    let mut strings = [0; 4096];
    let mut offsets = [(0, 0); GROUP];
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
    let mut descriptor_keys = [dummy; 128];
    let mut descriptor_output = [0; IO_BYTES];
    let mut descriptor_input = [0; IO_BYTES];
    let mut encoding = [0; 16 * 1024];
    let mut directory = DirectoryScratch::new();
    let mut directory_encoding = [0; ENCODING_BYTES];
    let shape = VolumeShape {
        layout: VolumeLayout::RowId,
        row_count: ROWS as u64,
        column_count: 2,
        group_count: ROWS.div_ceil(GROUP) as u64,
        rows: Some(RowBounds {
            min: -512,
            max: 512,
        }),
        window: None,
    };
    let page_limits = ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: 1024 * 1024,
        page_decoded_bytes: 1024 * 1024,
    };
    crc32fast::hash(b"warm CPU dispatch before measuring");
    CALLS.with(|calls| calls.set(0));
    TRACK.with(|track| track.set(true));
    let stop = Stop;
    let mut writer = SpoolWriter::new(&mut payload, binding, limits, &mut write_buffer).unwrap();
    let mut runs =
        RowRunWriter::new(&mut first, binding, limits, &mut row_keys, &mut run_output).unwrap();
    for index in (0..ROWS).rev() {
        let id = index as i64 - 512;
        values[0] = Value::Integer(id * 13);
        let receipt = writer
            .append(CapturedRow {
                row_id: id,
                creator_txn_id: index as i64 + 1,
                source: RowSource::Dml(NonZeroU64::new(index as u64 + 7).unwrap()),
                values: &values,
            })
            .unwrap();
        runs.push(receipt).unwrap();
    }
    let summary = writer.finish().unwrap();
    let mut runs = runs.finish(summary).unwrap();
    while !runs.is_sorted() {
        runs = merge_pass(
            &FileSource::new(&first),
            &mut second,
            runs,
            &mut left,
            &mut right,
            &mut merge_output,
        )
        .unwrap();
        std::mem::swap(&mut first, &mut second);
    }
    let sorted_source = FileSource::new(&first);
    let mut sorted = SortedRows::new(&sorted_source, runs, &mut sorted_input).unwrap();
    let payload_source = FileSource::new(&payload);
    let mut gather = SpoolReader::new(&payload_source, summary, &mut read_window).unwrap();
    let mut metadata = RunWriter::new(
        &mut descriptors,
        &mut descriptor_keys,
        &mut descriptor_output,
    )
    .unwrap();
    let mut writer = PayloadWriter::new(
        &mut volume,
        header,
        shape,
        None,
        &columns,
        page_limits,
        ColumnLimits::default(),
        &mut metadata,
    )
    .unwrap();
    for start in (0..ROWS).step_by(GROUP) {
        let count = GROUP.min(ROWS - start);
        for local in 0..count {
            positions[local] = sorted.next_position().unwrap().unwrap();
            let row = positions[local];
            let id = (start + local) as i64 - 512;
            assert_eq!(row.row_id(), id);
            assert_eq!(row.creator_txn_id(), (start + local) as i64 + 1);
            assert_eq!(
                row.source(),
                RowSource::Dml(NonZeroU64::new((start + local) as u64 + 7).unwrap())
            );
            row_ids[local] = id;
            row_sources[local] = row.source();
        }
        writer
            .begin_group(
                GroupRecord {
                    row_start: start as u64,
                    row_count: count as u32,
                    column_count: 2,
                    rows: RowBounds {
                        min: row_ids[0],
                        max: row_ids[count - 1],
                    },
                },
                &row_ids[..count],
                &row_sources[..count],
                &mut encoding,
            )
            .unwrap();
        let mut group = gather.prepare_group(&positions[..count]).unwrap();
        let column = group
            .gather_column(0, &mut spans, &mut nulls, ColumnBuffer::I64(&mut integers))
            .unwrap();
        let ColumnInput::I64(actual) = column.input else {
            panic!()
        };
        for (actual, id) in actual.iter().zip(&row_ids) {
            assert_eq!(*actual, id * 13);
        }
        writer
            .write_column(column.nulls, column.input, &mut encoding)
            .unwrap();
        let column = group
            .gather_column(
                1,
                &mut spans,
                &mut nulls,
                ColumnBuffer::Variable {
                    bytes: &mut strings,
                    offsets: &mut offsets,
                },
            )
            .unwrap();
        let ColumnInput::Variable { data, offsets } = column.input else {
            panic!()
        };
        for &(offset, len) in offsets {
            assert_eq!(
                &data[offset as usize..(offset + len) as usize],
                "captured λ payload".as_bytes()
            );
        }
        writer
            .write_column(column.nulls, column.input, &mut encoding)
            .unwrap();
    }
    assert!(sorted.next_position().unwrap().is_none());
    let mut completed = writer.finish_payloads().unwrap();
    let metadata = metadata.finish().unwrap();
    let descriptor_source = FileSource::new(&descriptors);
    let mut sorted =
        SortedReader::new(&descriptor_source, metadata, &mut descriptor_input).unwrap();
    let finished = completed
        .finish(
            || sorted.next_entry(),
            &mut directory,
            &mut directory_encoding,
        )
        .unwrap();
    std::hint::black_box(finished);
    drop(stop);
    assert_eq!(CALLS.with(Cell::get), 0);
    assert_eq!(payload.written, summary.byte_len());
    assert_eq!(
        payload.writes as u64,
        summary.byte_len().div_ceil(write_buffer.len() as u64)
    );
    eprintln!(
        "captured spool: {} File write calls / {} bytes for {} rows; caller write buffer={} bytes",
        payload.writes,
        payload.written,
        ROWS,
        write_buffer.len()
    );
    eprintln!("reverse payload gather: {} read calls, {} bytes, {} spool bytes; caller window=8192 bytes; allocations=0",
        payload_source.calls.get(), payload_source.bytes.get(), summary.byte_len());
    // Reopen the actual completed file independently, validate its envelope,
    // look up every expected directory key, and decode every emitted page. This
    // verification is outside the producer allocation meter and never reads
    // the complete file into a Vec.
    let source = FileSource::new(&volume);
    let opened =
        OpenedEnvelope::read(&source, volume.metadata().unwrap().len(), &page_limits).unwrap();
    opened.require_identity(header.identity).unwrap();
    let mut root_bytes = [0; 128];
    let root = PageReadPlan::for_root(&opened.footer, &page_limits)
        .unwrap()
        .read_into(&source, &mut root_bytes, &mut [])
        .unwrap();
    let root = RootSummary::decode(root, &opened.footer, &page_limits).unwrap();
    assert_eq!(root, finished.summary);
    let source_context = SourceContext::bind(&opened.header, &root, None).unwrap();
    let mut nodes = [0; MAX_NODE_BYTES];
    let mut lookup = DirectoryLookup::new(
        &source,
        opened.footer,
        root,
        page_limits,
        &mut nodes,
        &mut [],
    )
    .unwrap();
    for group in 0..shape.group_count {
        let entry = lookup
            .find(KeyIdentity {
                section: Section::GroupMetadata as u16,
                column: GLOBAL_COLUMN,
                ordinal: group,
            })
            .unwrap()
            .unwrap();
        let bytes = PageReadPlan::for_page(&opened.footer, entry.page, &page_limits)
            .unwrap()
            .read_into(&source, &mut encoding, &mut [])
            .unwrap();
        let metadata = GroupPageExpectation::new(&root, group, 1)
            .unwrap()
            .decode(bytes)
            .unwrap();
        let expected = metadata.iter().next().unwrap();
        let record = expected.record();
        assert_eq!(record.row_start, group * GROUP as u64);
        let identity = IdentityPageExpectation::new(expected);
        let entry = lookup
            .find(KeyIdentity {
                section: Section::RowIds as u16,
                column: GLOBAL_COLUMN,
                ordinal: group,
            })
            .unwrap()
            .unwrap();
        let bytes = identity
            .read_plan(&opened.footer, entry.page, &page_limits)
            .unwrap()
            .read_into(&source, &mut encoding, &mut [])
            .unwrap();
        let ids = identity.decode_row_ids(bytes).unwrap();
        for local in 0..record.row_count as usize {
            assert_eq!(
                ids.get(local),
                Some(record.row_start as i64 + local as i64 - 512)
            );
        }
        let entry = lookup
            .find(KeyIdentity {
                section: Section::SourceLsns as u16,
                column: GLOBAL_COLUMN,
                ordinal: group,
            })
            .unwrap()
            .unwrap();
        let bytes = identity
            .read_plan(&opened.footer, entry.page, &page_limits)
            .unwrap()
            .read_into(&source, &mut encoding, &mut [])
            .unwrap();
        let sources = identity.decode_sources(bytes, source_context).unwrap();
        for local in 0..record.row_count as usize {
            assert_eq!(
                sources.get(local),
                Some(RowSource::Dml(
                    NonZeroU64::new(record.row_start + local as u64 + 7).unwrap()
                ))
            );
        }
        for (column, spec) in columns.iter().enumerate() {
            let entry = lookup
                .find(KeyIdentity {
                    section: Section::ColumnBlocks as u16,
                    column: column as u32,
                    ordinal: group,
                })
                .unwrap()
                .unwrap();
            let bytes = PageReadPlan::for_page(&opened.footer, entry.page, &page_limits)
                .unwrap()
                .read_into(&source, &mut encoding, &mut [])
                .unwrap();
            let expected = expected
                .column(column as u32, spec.data_type, spec.vector_dimensions)
                .unwrap();
            let block = ColumnBlockRef::parse(bytes, expected, ColumnLimits::default()).unwrap();
            for local in 0..record.row_count as usize {
                let value = if column == 0 {
                    ColumnCell::Integer((record.row_start as i64 + local as i64 - 512) * 13)
                } else {
                    ColumnCell::Text("captured λ payload")
                };
                assert_eq!(block.cell(local), Some(value));
            }
        }
    }
}
