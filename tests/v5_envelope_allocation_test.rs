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

use stoolap::storage::volume::v5::envelope::{
    Codec, EnvelopeError, FileIdentity, Footer, Header, PageDescriptor, ReadLimits,
};

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountAlloc;
fn record_allocation() {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            ALLOCATIONS.with(|count| count.set(count.get() + 1));
        }
    });
}
// Test-only counting shim: every allocation is delegated unchanged to System.
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

struct StopTracking;
impl Drop for StopTracking {
    fn drop(&mut self) {
        TRACK.with(|track| track.set(false));
    }
}

#[test]
fn streaming_directory_retains_fixed_caller_scratch_without_allocating() {
    use stoolap::storage::volume::v5::directory::{
        DirectoryKey, Layout as VolumeLayout, LeafEntry, RootSummary, RowBounds, Section,
        KEY_REQUIRED,
    };
    use stoolap::storage::volume::v5::directory_writer::{
        DirectoryScratch, DirectoryWriter, ENCODING_BYTES,
    };
    use stoolap::storage::volume::v5::page_io::PageWriter;

    let limits = ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: ENCODING_BYTES as u64,
        page_decoded_bytes: ENCODING_BYTES as u64,
    };
    let header = Header::new(FileIdentity::new(1, 1, 1).unwrap());
    assert!(std::mem::size_of::<DirectoryScratch>() + ENCODING_BYTES <= 36 * 1024);
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    {
        let _stop = StopTracking;
        let mut scratch = DirectoryScratch::new();
        let mut encoding = [0; ENCODING_BYTES];
        for count in [1, 65, 10_000] {
            let mut sink = std::io::sink();
            let mut writer = PageWriter::new(&mut sink, header, limits).unwrap();
            let page = writer.append_stored(Codec::Raw, b"x", 1).unwrap();
            let built = {
                let mut builder =
                    DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding).unwrap();
                let mut entry = LeafEntry {
                    key: DirectoryKey {
                        section: Section::RowIds as u16,
                        flags: KEY_REQUIRED,
                        column: u32::MAX,
                        ordinal: 0,
                    },
                    page,
                };
                for ordinal in 0..count {
                    entry.key.ordinal = ordinal;
                    builder.push(black_box(entry)).unwrap();
                }
                // An invalid duplicate has no allocation either and preserves
                // the accepted stream so completion remains possible.
                assert!(builder.push(black_box(entry)).is_err());
                builder.finish().unwrap()
            };
            let summary = RootSummary {
                layout: VolumeLayout::RowId,
                legacy_base: None,
                row_count: 1,
                column_count: 1,
                group_count: 1,
                entry_count: built.entry_count,
                rows: Some(RowBounds { min: 1, max: 1 }),
                window: None,
                directory: built.root,
            };
            black_box(writer.finish(&summary).unwrap());
        }
    }
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}

#[test]
fn envelope_success_and_corruption_paths_allocate_nothing() {
    let limits = ReadLimits {
        root_stored_bytes: 64,
        root_decoded_bytes: 64,
        page_stored_bytes: 64,
        page_decoded_bytes: 64,
    };
    let header = Header::new(FileIdentity::new(1, 1, 1).unwrap());
    let page = PageDescriptor {
        offset: 64,
        stored_len: 9,
        decoded_len: 9,
        stored_checksum: 0xcbf4_3926,
        codec: Codec::Raw,
    };
    let footer = Footer {
        file_length: 169,
        root: PageDescriptor { offset: 96, ..page },
    };
    let header_bytes = header.encode().unwrap();
    let footer_bytes = footer.encode().unwrap();
    let page_bytes = page.encode().unwrap();
    let mut corrupted = header_bytes;
    corrupted[8] ^= 1;
    // Initialize TLS before measurement; unrelated test threads are excluded.
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    {
        let _stop = StopTracking;
        for _ in 0..1024 {
            black_box(Header::decode(black_box(&header_bytes)).unwrap());
            black_box(header.encode().unwrap());
            let decoded = Footer::decode(black_box(&footer_bytes), 169, &limits).unwrap();
            black_box(footer.encode().unwrap());
            black_box(PageDescriptor::decode(black_box(&page_bytes)).unwrap());
            decoded.validate_page(black_box(&page), &limits).unwrap();
            page.verify_stored_bytes(black_box(b"123456789")).unwrap();
            assert_eq!(
                Header::decode(black_box(&corrupted)),
                Err(EnvelopeError::ChecksumMismatch)
            );
            assert_eq!(
                Footer::decode(black_box(&footer_bytes), 170, &limits),
                Err(EnvelopeError::FileLengthMismatch)
            );
            assert_eq!(
                page.verify_stored_bytes(black_box(b"123456788")),
                Err(EnvelopeError::ChecksumMismatch)
            );
            assert!(FileIdentity::new(0, 1, 1).is_err());
        }
    }
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}

#[test]
fn directory_success_and_corruption_paths_allocate_nothing() {
    use stoolap::storage::volume::v5::directory::{
        encode_interior, encode_leaf, DirectoryError, DirectoryKey, DirectoryNode, DirectoryRoot,
        InteriorEntry, Layout as VolumeLayout, LeafEntry, RootSummary, RowBounds, Section,
        WindowBounds, KEY_REQUIRED, MAX_FANOUT, MAX_NODE_BYTES,
    };
    let limits = ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: MAX_NODE_BYTES as u64,
        page_decoded_bytes: MAX_NODE_BYTES as u64,
    };
    let raw_page = |offset, len| PageDescriptor {
        offset,
        stored_len: len,
        decoded_len: len,
        stored_checksum: 0,
        codec: Codec::Raw,
    };
    let footer = Footer {
        file_length: 32_768 + 128 + 64,
        root: raw_page(32_768, 128),
    };
    let entries: [LeafEntry; MAX_FANOUT] = std::array::from_fn(|i| LeafEntry {
        key: DirectoryKey {
            section: Section::SourceLsns as u16,
            flags: KEY_REQUIRED,
            column: u32::MAX,
            ordinal: i as u64,
        },
        page: raw_page(64, 8),
    });
    let mut leaf_bytes = [0; MAX_NODE_BYTES];
    let leaf_len = encode_leaf(&entries, &mut leaf_bytes).unwrap();
    let leaf_page = raw_page(4096, leaf_len as u64);
    let interior_entry = InteriorEntry {
        lower: entries[0].key,
        upper: entries[MAX_FANOUT - 1].key,
        child: leaf_page,
    };
    let mut parent_bytes = [0; 96];
    encode_interior(2, MAX_FANOUT as u64, &[interior_entry], &mut parent_bytes).unwrap();
    let parent_page = raw_page(16_384, 96);
    let summary = RootSummary {
        layout: VolumeLayout::RowId,
        legacy_base: None,
        row_count: MAX_FANOUT as u64,
        column_count: 3,
        group_count: 1,
        entry_count: MAX_FANOUT as u64,
        rows: Some(RowBounds {
            min: i64::MIN,
            max: i64::MAX,
        }),
        window: Some(WindowBounds { lower: 3, upper: 3 }),
        directory: Some(DirectoryRoot {
            depth: 2,
            page: parent_page,
        }),
    };
    let summary_bytes = summary.encode().unwrap();
    let mut bad_leaf = leaf_bytes;
    // Duplicate key in the last entry makes decode validate the preceding 63
    // entries before failing, including their descriptors and backward bounds.
    let last_ordinal = 32 + (MAX_FANOUT - 1) * 48 + 8;
    bad_leaf[last_ordinal..last_ordinal + 8].copy_from_slice(&0u64.to_le_bytes());
    let mut bad_summary = summary_bytes;
    bad_summary[112] = 1;
    let mut output = [0; MAX_NODE_BYTES];
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    {
        let _stop = StopTracking;
        for _ in 0..1024 {
            let child = DirectoryNode::decode(
                black_box(&leaf_bytes[..leaf_len]),
                leaf_page,
                &footer,
                &limits,
            )
            .unwrap();
            let parent =
                DirectoryNode::decode(black_box(&parent_bytes), parent_page, &footer, &limits)
                    .unwrap();
            parent.validate_child(0, &child).unwrap();
            for entry in child.leaves() {
                black_box(entry.unwrap());
            }
            for entry in parent.children() {
                black_box(entry.unwrap());
            }
            let decoded_root =
                RootSummary::decode(black_box(&summary_bytes), &footer, &limits).unwrap();
            decoded_root.validate_directory_root(&parent).unwrap();
            black_box(summary.encode().unwrap());
            black_box(encode_leaf(&entries, &mut output).unwrap());
            black_box(
                encode_interior(2, MAX_FANOUT as u64, &[interior_entry], &mut output).unwrap(),
            );
            assert_eq!(
                DirectoryNode::decode(
                    black_box(&bad_leaf[..leaf_len]),
                    leaf_page,
                    &footer,
                    &limits
                )
                .unwrap_err(),
                DirectoryError::KeyOrder
            );
            assert_eq!(
                RootSummary::decode(black_box(&bad_summary), &footer, &limits),
                Err(DirectoryError::Reserved)
            );
        }
    }
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}

#[test]
fn page_io_uses_only_caller_buffers_for_success_and_errors() {
    use stoolap::storage::volume::v5::directory::{
        DirectoryKey, DirectoryRoot, Layout as VolumeLayout, LeafEntry, RootSummary, RowBounds,
        Section, KEY_REQUIRED,
    };
    use stoolap::storage::volume::v5::page_io::{
        OpenedEnvelope, PageIoError, PageReadPlan, PageWriter, ReadAt,
    };
    struct Slice<'a>(&'a [u8]);
    impl ReadAt for Slice<'_> {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> std::io::Result<usize> {
            let Some(bytes) = usize::try_from(offset).ok().and_then(|p| self.0.get(p..)) else {
                return Ok(0);
            };
            let n = bytes.len().min(dst.len()).min(7);
            dst[..n].copy_from_slice(&bytes[..n]);
            Ok(n)
        }
    }
    let limits = ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: 8192,
        page_decoded_bytes: 8192,
    };
    let identity = FileIdentity::new(1, 1, 2).unwrap();
    let header = Header::new(identity);
    // Explicit LZ4 block containing five literal bytes, independent of encoder.
    let compressed = [0x50, b'h', b'e', b'l', b'l', b'o'];
    let mut file = [0u8; 512];
    let mut scratch = [0; 80];
    let mut stored = [0; 128];
    let mut decoded = [0; 128];
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    {
        let _stop = StopTracking;
        for _ in 0..1024 {
            let mut sink = file.as_mut_slice();
            let mut writer = PageWriter::new(&mut sink, header, limits).unwrap();
            let payload = writer
                .append_stored(Codec::Lz4Block, &compressed, 5)
                .unwrap();
            let leaf = writer
                .append_leaf(
                    &[LeafEntry {
                        key: DirectoryKey {
                            section: Section::SourceLsns as u16,
                            flags: KEY_REQUIRED,
                            column: u32::MAX,
                            ordinal: 0,
                        },
                        page: payload,
                    }],
                    &mut scratch,
                )
                .unwrap();
            let summary = RootSummary {
                layout: VolumeLayout::RowId,
                legacy_base: None,
                row_count: 1,
                column_count: 0,
                group_count: 1,
                entry_count: 1,
                rows: Some(RowBounds { min: 0, max: 0 }),
                window: None,
                directory: Some(DirectoryRoot {
                    depth: 1,
                    page: leaf,
                }),
            };
            let finished = writer.finish(&summary).unwrap();
            assert!(matches!(
                writer.append_stored(Codec::Raw, b"x", 1),
                Err(PageIoError::Finished)
            ));
            let source = Slice(&file[..finished.footer.file_length as usize]);
            let opened =
                OpenedEnvelope::read(&source, finished.footer.file_length, &limits).unwrap();
            opened.require_identity(identity).unwrap();
            let root_plan = PageReadPlan::for_root(&opened.footer, &limits).unwrap();
            let root_bytes = root_plan
                .read_into(&source, &mut stored, &mut decoded)
                .unwrap();
            assert_eq!(
                RootSummary::decode(root_bytes, &opened.footer, &limits).unwrap(),
                summary
            );
            let payload_plan = PageReadPlan::for_page(&opened.footer, payload, &limits).unwrap();
            assert_eq!(
                payload_plan
                    .read_into(&source, &mut stored, &mut decoded)
                    .unwrap(),
                b"hello"
            );
            let corrupt = PageDescriptor {
                stored_checksum: payload.stored_checksum ^ 1,
                ..payload
            };
            let bad_plan = PageReadPlan::for_page(&opened.footer, corrupt, &limits).unwrap();
            assert!(matches!(
                bad_plan.read_into(&source, &mut stored, &mut decoded),
                Err(PageIoError::Envelope(EnvelopeError::ChecksumMismatch))
            ));
            assert!(matches!(
                payload_plan.read_into(&Slice(&[]), &mut stored, &mut decoded),
                Err(PageIoError::UnexpectedEof)
            ));
            let mut too_short = [0u8; 2];
            let mut sink = too_short.as_mut_slice();
            assert!(matches!(
                PageWriter::new(&mut sink, header, limits),
                Err(PageIoError::Io(_))
            ));
        }
    }
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}

#[test]
fn directory_lookup_and_structural_walk_allocate_nothing() {
    use stoolap::storage::volume::v5::directory::{
        DirectoryKey, Layout as VolumeLayout, LeafEntry, RootSummary, RowBounds, Section,
        KEY_REQUIRED, MAX_NODE_BYTES,
    };
    use stoolap::storage::volume::v5::directory_reader::{
        DirectoryBufferRequirements, DirectoryLookup, DirectoryWalker, KeyIdentity,
    };
    use stoolap::storage::volume::v5::directory_writer::{
        DirectoryScratch, DirectoryWriter, ENCODING_BYTES,
    };
    use stoolap::storage::volume::v5::page_io::{PageWriter, ReadAt};
    struct Source<'a> {
        bytes: &'a [u8],
        fail: Cell<bool>,
        calls: Cell<usize>,
    }
    impl ReadAt for Source<'_> {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> std::io::Result<usize> {
            self.calls.set(self.calls.get() + 1);
            if self.fail.get() {
                return Err(std::io::ErrorKind::PermissionDenied.into());
            }
            let Some(bytes) = usize::try_from(offset)
                .ok()
                .and_then(|p| self.bytes.get(p..))
            else {
                return Ok(0);
            };
            let n = bytes.len().min(dst.len());
            dst[..n].copy_from_slice(&bytes[..n]);
            Ok(n)
        }
    }
    let limits = ReadLimits {
        root_stored_bytes: 128,
        root_decoded_bytes: 128,
        page_stored_bytes: 8192,
        page_decoded_bytes: 8192,
    };
    let header = Header::new(FileIdentity::new(1, 1, 1).unwrap());
    for count in [1u64, 65, 10_000] {
        // File construction and reservation are outside the measured read phase.
        let mut file = Vec::new();
        let mut writer = PageWriter::new(&mut file, header, limits).unwrap();
        let page = writer.append_stored(Codec::Raw, b"x", 1).unwrap();
        let mut scratch = DirectoryScratch::new();
        let mut encoding = [0; ENCODING_BYTES];
        let mut builder = DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding).unwrap();
        for ordinal in 0..count {
            builder
                .push(LeafEntry {
                    key: DirectoryKey {
                        section: Section::RowIds as u16,
                        flags: KEY_REQUIRED,
                        column: u32::MAX,
                        ordinal,
                    },
                    page,
                })
                .unwrap();
        }
        let built = builder.finish().unwrap();
        let summary = RootSummary {
            layout: VolumeLayout::RowId,
            legacy_base: None,
            row_count: 1,
            column_count: 0,
            group_count: 1,
            entry_count: count,
            rows: Some(RowBounds { min: 0, max: 0 }),
            window: None,
            directory: built.root,
        };
        let finished = writer.finish(&summary).unwrap();
        let source = Source {
            bytes: &file,
            fail: Cell::new(false),
            calls: Cell::new(0),
        };
        let requirements =
            DirectoryBufferRequirements::new(&finished.footer, &summary, &limits).unwrap();
        let mut lookup_node = [0; MAX_NODE_BYTES];
        let mut nodes = vec![[0; MAX_NODE_BYTES]; requirements.walker_slots];
        ALLOCATIONS.with(|count| count.set(0));
        TRACK.with(|track| track.set(true));
        {
            let _stop = StopTracking;
            let mut lookup = DirectoryLookup::new(
                &source,
                finished.footer,
                summary,
                limits,
                &mut lookup_node,
                &mut [],
            )
            .unwrap();
            let key = KeyIdentity {
                section: Section::RowIds as u16,
                column: u32::MAX,
                ordinal: count - 1,
            };
            assert!(lookup.find(black_box(key)).unwrap().is_some());
            assert!(lookup
                .find(KeyIdentity {
                    ordinal: count,
                    ..key
                })
                .unwrap()
                .is_none());
            let mut walker = DirectoryWalker::new(
                &source,
                finished.footer,
                summary,
                limits,
                &mut nodes,
                &mut [],
            )
            .unwrap();
            for ordinal in 0..count.min(10) {
                assert_eq!(walker.next_entry().unwrap().unwrap().key.ordinal, ordinal);
            }
            assert_eq!(walker.finish_validation().unwrap().entry_count, count);
            assert!(walker.next_entry().unwrap().is_none());
            source.fail.set(true);
            assert!(lookup.find(key).is_err());
            let calls = source.calls.get();
            assert!(lookup.find(key).is_err());
            assert_eq!(source.calls.get(), calls);
        }
        assert_eq!(ALLOCATIONS.with(Cell::get), 0);
    }
}

#[test]
fn group_and_identity_codecs_allocate_nothing_at_full_bounds_and_on_corruption() {
    use std::num::NonZeroU64;
    use stoolap::storage::volume::v5::directory::{
        DirectoryRoot, Layout as VolumeLayout, RootSummary, RowBounds,
    };
    use stoolap::storage::volume::v5::envelope::{LegacyBase, REQUIRED_LEGACY_BASE};
    use stoolap::storage::volume::v5::group_metadata::{
        GroupPageExpectation, GroupRangeValidator, GroupRecord, MAX_GROUP_PAGE_BYTES,
    };
    use stoolap::storage::volume::v5::row_identity::{
        IdentityPageExpectation, RowSource, SourceContext, VerifiedLegacyBase,
        MAX_IDENTITY_PAGE_BYTES,
    };

    let identity = FileIdentity::new(1, 1, 1).unwrap();
    let base = LegacyBase {
        generation: NonZeroU64::new(7).unwrap(),
        barrier_lsn: 100,
    };
    let root = RootSummary {
        layout: VolumeLayout::RowId,
        legacy_base: Some(base),
        row_count: 64 * 4096,
        column_count: 2,
        group_count: 64,
        entry_count: 1,
        rows: Some(RowBounds {
            min: 0,
            max: 64 * 4096 - 1,
        }),
        window: None,
        directory: Some(DirectoryRoot {
            depth: 1,
            page: PageDescriptor {
                offset: 64,
                stored_len: 80,
                decoded_len: 80,
                stored_checksum: 0,
                codec: Codec::Raw,
            },
        }),
    };
    let mut header = Header::new(identity);
    header.required_features |= REQUIRED_LEGACY_BASE;
    let evidence = VerifiedLegacyBase::assert_verified_checkpoint(identity, base);
    let records = std::array::from_fn::<_, 64, _>(|i| GroupRecord {
        row_start: i as u64 * 4096,
        row_count: 4096,
        column_count: 2,
        rows: RowBounds {
            min: i as i64 * 4096,
            max: (i as i64 + 1) * 4096 - 1,
        },
    });
    let row_ids = std::array::from_fn::<_, 4096, _>(|i| i as i64);
    let sources = std::array::from_fn::<_, 4096, _>(|i| {
        if i % 2 == 0 {
            RowSource::LegacyBase
        } else {
            RowSource::Dml(NonZeroU64::new(101).unwrap())
        }
    });
    let mut group_bytes = [0; MAX_GROUP_PAGE_BYTES];
    let mut row_bytes = [0; MAX_IDENTITY_PAGE_BYTES];
    let mut source_bytes = [0; MAX_IDENTITY_PAGE_BYTES];
    let mut converted_source_bytes = [0; MAX_IDENTITY_PAGE_BYTES];
    ALLOCATIONS.with(|count| count.set(0));
    TRACK.with(|track| track.set(true));
    {
        let _stop = StopTracking;
        let context = SourceContext::bind(&header, &root, Some(&evidence)).unwrap();
        let expected = GroupPageExpectation::new(&root, 0, 64).unwrap();
        expected.encode(&records, &mut group_bytes).unwrap();
        let groups = expected.decode(&group_bytes).unwrap();
        let mut coverage = GroupRangeValidator::new(&root).unwrap();
        for group in groups.iter() {
            coverage.push(group).unwrap();
        }
        coverage.finish().unwrap();
        let group = groups.iter().next().unwrap();
        black_box(
            group
                .column(1, stoolap::core::DataType::Integer, None)
                .unwrap(),
        );
        let expected_ids = IdentityPageExpectation::new(group);
        expected_ids
            .encode_row_ids(&row_ids, &mut row_bytes)
            .unwrap();
        let ids = expected_ids.decode_row_ids(&row_bytes).unwrap();
        assert!(ids.iter().eq(row_ids));
        expected_ids
            .encode_sources(&sources, context, &mut source_bytes)
            .unwrap();
        assert!(expected_ids
            .decode_sources(&source_bytes, context)
            .unwrap()
            .iter()
            .eq(sources));
        let decoded_sources = expected_ids.decode_sources(&source_bytes, context).unwrap();
        expected_ids
            .encode_sources_from_page(decoded_sources, context, &mut converted_source_bytes)
            .unwrap();
        assert_eq!(source_bytes, converted_source_bytes);
        let plain = SourceContext::bind(
            &Header::new(identity),
            &RootSummary {
                legacy_base: None,
                ..root
            },
            None,
        )
        .unwrap();
        let fresh = [RowSource::Dml(NonZeroU64::new(101).unwrap()); 4096];
        expected_ids
            .encode_sources(&fresh, plain, &mut source_bytes)
            .unwrap();
        let fresh_page = expected_ids.decode_sources(&source_bytes, plain).unwrap();
        expected_ids
            .encode_sources_from_page(fresh_page, context, &mut converted_source_bytes)
            .unwrap();
        assert!(expected_ids
            .decode_sources(&converted_source_bytes, context)
            .unwrap()
            .iter()
            .eq(fresh));
        source_bytes[MAX_IDENTITY_PAGE_BYTES - 8..].copy_from_slice(&100u64.to_le_bytes());
        assert!(expected_ids.decode_sources(&source_bytes, context).is_err());
        row_bytes[MAX_IDENTITY_PAGE_BYTES - 8..].copy_from_slice(&0i64.to_le_bytes());
        assert!(expected_ids.decode_row_ids(&row_bytes).is_err());
        group_bytes[MAX_GROUP_PAGE_BYTES - 32 + 12] = 1;
        assert!(expected.decode(&group_bytes).is_err());
        assert!(SourceContext::bind(&header, &root, None).is_err());
    }
    assert_eq!(ALLOCATIONS.with(Cell::get), 0);
}
