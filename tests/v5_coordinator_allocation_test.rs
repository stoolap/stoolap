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

use lz4_flex::block::CompressTable;
use std::fs::File;
use stoolap::core::{DataType, Value};
use stoolap::storage::volume::v5::column_block::{ColumnBlockRef, ColumnCell, ColumnLimits};
use stoolap::storage::volume::v5::compression::CompressionPlan;
use stoolap::storage::volume::v5::directory::{
    DirectoryKey, LeafEntry, Section, GLOBAL_COLUMN, KEY_REQUIRED,
};
use stoolap::storage::volume::v5::directory::{RootSummary, MAX_NODE_BYTES};
use stoolap::storage::volume::v5::directory_reader::{DirectoryLookup, KeyIdentity};
use stoolap::storage::volume::v5::directory_writer::{DirectoryScratch, ENCODING_BYTES};
use stoolap::storage::volume::v5::envelope::{
    Codec, FileIdentity, Header, PageDescriptor, ReadLimits,
};
use stoolap::storage::volume::v5::group_metadata::GroupPageExpectation;
use stoolap::storage::volume::v5::metadata_runs::{RunWriter, SortedReader, IO_BYTES};
use stoolap::storage::volume::v5::page_io::{OpenedEnvelope, PageReadPlan, ReadAt};
use stoolap::storage::volume::v5::payload_writer::ColumnSpec;
use stoolap::storage::volume::v5::row_identity::RowSource;
use stoolap::storage::volume::v5::row_identity::SourceEncodingContext;
use stoolap::storage::volume::v5::row_identity::{IdentityPageExpectation, SourceContext};
use stoolap::storage::volume::v5::row_spool::runs::{merge_pass, RowRunWriter};
use stoolap::storage::volume::v5::row_spool::{
    CapturedRow, CellSpan, RowPosition, SpoolBinding, SpoolLimits, SpoolWriter,
};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static CALLS: Cell<usize> = const { Cell::new(0) };
    static REQUESTED: Cell<usize> = const { Cell::new(0) };
}
struct Counting;
fn count(bytes: usize) {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            CALLS.with(|calls| calls.set(calls.get() + 1));
            REQUESTED.with(|requested| requested.set(requested.get() + bytes));
        }
    });
}
// Test-only shim forwards the exact allocation/deallocation to System.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, n: usize) -> *mut u8 {
        count(n);
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

use stoolap::storage::volume::v5::coordinator::{
    emit_payloads, plan_groups, BuildConfig, ColumnScratch, CompressionScratch, EmissionScratch,
    PlanningScratch,
};
use stoolap::storage::volume::v5::metadata_runs::{merge_pass as merge_descriptors, MergeScratch};

#[test]
fn files_stream_through_coordinator_within_16_mib_caller_scratch() {
    pipeline(16 * 1024 * 1024, 512, false);
}
#[test]
fn files_stream_through_coordinator_within_64_mib_caller_scratch() {
    pipeline(64 * 1024 * 1024, 4096, false);
}

#[test]
fn files_compress_through_coordinator_within_16_mib_caller_scratch() {
    pipeline(16 * 1024 * 1024, 512, true);
}

fn pipeline(scratch_limit: usize, batch: usize, compress: bool) {
    const ROWS: usize = 8193;
    const CAP: usize = 1024 * 1024;
    // File opening, schema, a SINGLE reusable captured value, and all caller
    // buffers are outside the meter. No all-row or all-column payload exists.
    let mut payload = CountedFile {
        file: tempfile::tempfile().unwrap(),
        writes: 0,
        written: 0,
    };
    let mut first = tempfile::tempfile().unwrap();
    let mut second = tempfile::tempfile().unwrap();
    let mut boundary = tempfile::tempfile().unwrap();
    let mut volume = tempfile::tempfile().unwrap();
    let mut descriptors = tempfile::tempfile().unwrap();
    let mut descriptor_other = tempfile::tempfile().unwrap();
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
    let large = Value::from("λ".repeat(4096));
    let mut values = [Value::Integer(0), large.clone()];
    let mut row_keys = vec![RowPosition::EMPTY; 1024];
    let mut write_buffer = vec![0; 65536];
    let mut run_output = vec![0; 65536];
    let (mut left, mut right, mut merge_output) = (vec![0; 65536], vec![0; 65536], vec![0; 65536]);
    let mut positions = vec![RowPosition::EMPTY; batch];
    let mut costs = vec![0; batch];
    let mut row_ids = vec![0; batch];
    let mut sources = vec![RowSource::Dml(NonZeroU64::MIN); batch];
    let mut read_window = vec![0; 65536];
    let mut sorted_input = vec![0; 65536];
    let mut plan_io = vec![0; 65536];
    let mut spans = vec![CellSpan::EMPTY; batch];
    let mut nulls = vec![false; batch];
    let mut integers = vec![0; batch];
    let mut strings = vec![0; CAP];
    let mut offsets = vec![(0, 0); batch];
    let mut encoding = vec![0; CAP];
    let pages = ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: CAP as u64,
        page_decoded_bytes: CAP as u64,
    };
    let mut compressed = vec![
        0;
        if compress {
            CompressionPlan::new(CAP, &pages).unwrap().output_capacity()
        } else {
            0
        }
    ];
    // Measure the dependency's actual table reservation, then reset the meter
    // before the complete producer operation. No assumed table layout/size.
    REQUESTED.with(|n| n.set(0));
    CALLS.with(|n| n.set(0));
    TRACK.with(|t| t.set(true));
    let mut table = compress.then(CompressTable::large);
    TRACK.with(|t| t.set(false));
    let table_heap = REQUESTED.with(Cell::get);
    assert_eq!(CALLS.with(Cell::get), usize::from(compress));
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
    let mut descriptor_keys = vec![dummy; 128];
    let mut descriptor_output = vec![0; IO_BYTES];
    let mut descriptor_input = vec![0; IO_BYTES];
    let mut descriptor_merge = MergeScratch::new();
    let mut directory = DirectoryScratch::new();
    let mut directory_encoding = vec![0; ENCODING_BYTES];
    // Charge capacities, not current lengths. Native fixed objects are counted
    // as well; source/sink kernel buffers, input value and test verification are
    // outside this working-scratch contract, not an engine/RSS budget claim.
    let reserved = row_keys.capacity() * size_of::<RowPosition>()
        + positions.capacity() * size_of::<RowPosition>()
        + costs.capacity() * size_of::<u64>()
        + row_ids.capacity() * size_of::<i64>()
        + sources.capacity() * size_of::<RowSource>()
        + spans.capacity() * size_of::<CellSpan>()
        + nulls.capacity() * size_of::<bool>()
        + integers.capacity() * size_of::<i64>()
        + offsets.capacity() * size_of::<(u64, u64)>()
        + descriptor_keys.capacity() * size_of::<LeafEntry>()
        + [
            &write_buffer,
            &run_output,
            &left,
            &right,
            &merge_output,
            &read_window,
            &sorted_input,
            &plan_io,
            &strings,
            &encoding,
            &descriptor_output,
            &descriptor_input,
            &directory_encoding,
        ]
        .iter()
        .map(|v| v.capacity())
        .sum::<usize>()
        + size_of::<MergeScratch>()
        + size_of::<DirectoryScratch>()
        + compressed.capacity()
        + table_heap
        + size_of::<Option<CompressTable>>()
        + 4096; // reserved margin; compiler call-stack/RSS is outside this scratch bound
    assert!(reserved <= scratch_limit, "{reserved} > {scratch_limit}");
    let config = BuildConfig {
        rows: batch as u32,
        group_decoded_bytes: CAP,
        columns: ColumnLimits::default(),
        pages,
    };
    crc32fast::hash(b"warm dispatch");
    CALLS.with(|c| c.set(0));
    TRACK.with(|t| t.set(true));
    let stop = Stop;
    let mut writer = SpoolWriter::new(
        &mut payload,
        binding,
        SpoolLimits::default(),
        &mut write_buffer,
    )
    .unwrap();
    let mut runs = RowRunWriter::new(
        &mut first,
        binding,
        SpoolLimits::default(),
        &mut row_keys,
        &mut run_output,
    )
    .unwrap();
    for index in (0..ROWS).rev() {
        values[0] = Value::Integer(index as i64 * 13);
        values[1] = if index % 17 == 0 {
            Value::Null(DataType::Text)
        } else {
            large.clone()
        };
        runs.push(
            writer
                .append(CapturedRow {
                    row_id: index as i64 - 4096,
                    creator_txn_id: index as i64 + 1,
                    source: RowSource::Dml(NonZeroU64::new(index as u64 + 7).unwrap()),
                    values: &values,
                })
                .unwrap(),
        )
        .unwrap();
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
    let payload_source = FileSource::new(&payload);
    let plan = plan_groups(
        &sorted_source,
        &payload_source,
        &mut boundary,
        runs,
        config,
        &mut PlanningScratch {
            positions: &mut positions,
            prefix_costs: &mut costs,
            sorted_io: &mut sorted_input,
            payload_window: &mut read_window,
            plan_output: &mut plan_io,
            column: ColumnScratch {
                spans: &mut spans,
                nulls: &mut nulls,
                integers: &mut integers,
                floats: &mut [],
                booleans: &mut [],
                timestamps: &mut [],
                bytes: &mut strings,
                offsets: &mut offsets,
            },
        },
    )
    .unwrap();
    let shape = plan.shape();
    let planning_calls = payload_source.calls.get();
    let planning_bytes = payload_source.bytes.get();
    let mut metadata = RunWriter::new(
        &mut descriptors,
        &mut descriptor_keys,
        &mut descriptor_output,
    )
    .unwrap();
    let mut completed = emit_payloads(
        &sorted_source,
        &payload_source,
        &FileSource::new(&boundary),
        &mut volume,
        header,
        None,
        plan,
        &mut metadata,
        &mut EmissionScratch {
            positions: &mut positions,
            row_ids: &mut row_ids,
            sources: &mut sources,
            sorted_io: &mut sorted_input,
            payload_window: &mut read_window,
            plan_input: &mut plan_io,
            encoding: &mut encoding,
            compression: table.as_mut().map(|table| CompressionScratch {
                output: &mut compressed,
                table,
            }),
            column: ColumnScratch {
                spans: &mut spans,
                nulls: &mut nulls,
                integers: &mut integers,
                floats: &mut [],
                booleans: &mut [],
                timestamps: &mut [],
                bytes: &mut strings,
                offsets: &mut offsets,
            },
        },
    )
    .unwrap();
    let mut metadata = metadata.finish().unwrap();
    while !metadata.is_sorted() {
        metadata = merge_descriptors(
            &FileSource::new(&descriptors),
            &mut descriptor_other,
            metadata,
            &mut descriptor_merge,
        )
        .unwrap();
        std::mem::swap(&mut descriptors, &mut descriptor_other);
    }
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
    drop(stop);
    assert_eq!(CALLS.with(Cell::get), 0);
    assert!(summary.byte_len() > 32 * 1024 * 1024);
    assert_eq!(
        payload.writes as u64,
        summary.byte_len().div_ceil(write_buffer.len() as u64)
    );
    eprintln!(
        "coordinator compress={compress}, scratch cap={scratch_limit}, reserved={reserved}, table_heap={table_heap}, rows={ROWS}, groups={}, allocations=0; volume={} bytes; spool={} bytes / {} writes; planning={} reads / {} bytes; emission={} reads / {} bytes",
        shape.group_count,
        volume.metadata().unwrap().len(),
        summary.byte_len(),
        payload.writes,
        planning_calls,
        planning_bytes,
        payload_source.calls.get() - planning_calls,
        payload_source.bytes.get() - planning_bytes
    );
    // Reopen the actual completed file independently, validate its envelope,
    // look up every expected directory key, and decode every emitted page. This
    // verification is outside the producer allocation meter and never reads
    // the complete file into a Vec.
    let source = FileSource::new(&volume);
    let opened = OpenedEnvelope::read(&source, volume.metadata().unwrap().len(), &pages).unwrap();
    opened.require_identity(header.identity).unwrap();
    let mut root_bytes = [0; 128];
    let root = PageReadPlan::for_root(&opened.footer, &pages)
        .unwrap()
        .read_into(&source, &mut root_bytes, &mut [])
        .unwrap();
    let root = RootSummary::decode(root, &opened.footer, &pages).unwrap();
    assert_eq!(root, finished.summary);
    let source_context = SourceContext::bind(&opened.header, &root, None).unwrap();
    let mut nodes = [0; MAX_NODE_BYTES];
    let mut lookup =
        DirectoryLookup::new(&source, opened.footer, root, pages, &mut nodes, &mut []).unwrap();
    let mut seen = 0u64;
    let mut compressed_pages = 0;
    for group in 0..shape.group_count {
        let entry = lookup
            .find(KeyIdentity {
                section: Section::GroupMetadata as u16,
                column: GLOBAL_COLUMN,
                ordinal: group,
            })
            .unwrap()
            .unwrap();
        let bytes = PageReadPlan::for_page(&opened.footer, entry.page, &pages)
            .unwrap()
            .read_into(&source, &mut encoding, &mut [])
            .unwrap();
        let metadata = GroupPageExpectation::new(&root, group, 1)
            .unwrap()
            .decode(bytes)
            .unwrap();
        let expected = metadata.iter().next().unwrap();
        let record = expected.record();
        assert_eq!(record.row_start, seen);
        assert!(record.row_count as usize <= batch);
        seen += u64::from(record.row_count);
        let mut decoded_sum = 64u64;
        let identity = IdentityPageExpectation::new(expected);
        let entry = lookup
            .find(KeyIdentity {
                section: Section::RowIds as u16,
                column: GLOBAL_COLUMN,
                ordinal: group,
            })
            .unwrap()
            .unwrap();
        decoded_sum += entry.page.decoded_len;
        let bytes = identity
            .read_plan(&opened.footer, entry.page, &pages)
            .unwrap()
            .read_into(&source, &mut encoding, &mut [])
            .unwrap();
        let ids = identity.decode_row_ids(bytes).unwrap();
        for local in 0..record.row_count as usize {
            assert_eq!(
                ids.get(local),
                Some(record.row_start as i64 + local as i64 - 4096)
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
        decoded_sum += entry.page.decoded_len;
        let bytes = identity
            .read_plan(&opened.footer, entry.page, &pages)
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
            decoded_sum += entry.page.decoded_len;
            compressed_pages += usize::from(entry.page.codec == Codec::Lz4Block);
            let bytes = PageReadPlan::for_page(&opened.footer, entry.page, &pages)
                .unwrap()
                .read_into(&source, &mut encoding, &mut strings)
                .unwrap();
            let expected = expected
                .column(column as u32, spec.data_type, spec.vector_dimensions)
                .unwrap();
            let block = ColumnBlockRef::parse(bytes, expected, ColumnLimits::default()).unwrap();
            for local in 0..record.row_count as usize {
                let value = if column == 0 {
                    ColumnCell::Integer((record.row_start as i64 + local as i64) * 13)
                } else if (record.row_start + local as u64).is_multiple_of(17) {
                    ColumnCell::Null(DataType::Text)
                } else {
                    ColumnCell::Text(large.as_str().unwrap())
                };
                assert_eq!(block.cell(local), Some(value));
            }
        }
        assert!(decoded_sum <= CAP as u64, "group decoded {decoded_sum}");
    }

    assert_eq!(seen, ROWS as u64);
    assert_eq!(compressed_pages > 0, compress);
}
