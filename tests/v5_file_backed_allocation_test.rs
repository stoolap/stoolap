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
use std::fs::File;
use std::io::Write;
use std::num::NonZeroU64;

use stoolap::storage::volume::v5::directory::{
    DirectoryKey, DirectoryRoot, KEY_REQUIRED, Layout as VolumeLayout, LeafEntry, RowBounds,
    Section, VolumeShape,
};
use stoolap::storage::volume::v5::envelope::{Codec, FileIdentity, Header, ReadLimits};
use stoolap::storage::volume::v5::file_backed::FileBackedVolume;
use stoolap::storage::volume::v5::page_io::PageWriter;

thread_local! {
    static TRACK: Cell<bool> = const { Cell::new(false) };
    static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
}
struct Counting;
fn count(bytes: usize) {
    let _ = TRACK.try_with(|track| {
        if track.get() {
            COUNTS.with(|counter| {
                let (calls, total) = counter.get();
                counter.set((calls + 1, total + bytes));
            });
        }
    });
}
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Counting = Counting;
struct Stop;
impl Drop for Stop {
    fn drop(&mut self) {
        TRACK.with(|track| track.set(false));
    }
}
fn measured<T>(operation: impl FnOnce() -> T) -> (T, (usize, usize)) {
    COUNTS.with(|counter| counter.set((0, 0)));
    TRACK.with(|track| track.set(true));
    let stop = Stop;
    let output = operation();
    drop(stop);
    (output, COUNTS.with(Cell::get))
}

#[test]
fn large_file_open_retains_only_metadata_and_page_reads_do_not_allocate() {
    const PAGE: usize = 1 << 20;
    let dir = tempfile::tempdir().unwrap();
    let identity = FileIdentity {
        table_id: NonZeroU64::new(71).unwrap(),
        incarnation: NonZeroU64::new(72).unwrap(),
        volume_id: NonZeroU64::new(73).unwrap(),
    };
    let limits = ReadLimits {
        root_stored_bytes: 1024,
        root_decoded_bytes: 128,
        page_stored_bytes: PAGE as u64,
        page_decoded_bytes: PAGE as u64,
    };
    let mut page_scratch = vec![0; PAGE];
    let mut node = [0; 8192];
    let mut root_stored = [0; 1024];
    let mut root_decoded = [0; 128];
    let mut previous_open_bytes = None;
    for pages in [1, 32] {
        let path = dir.path().join(format!("size-{pages:02}.vol"));
        let mut file = File::create(&path).unwrap();
        let mut writer = PageWriter::new(&mut file, Header::new(identity), limits).unwrap();
        // Byte pages isolate ownership/IO behavior from typed row semantics.
        let mut entries = Vec::with_capacity(pages);
        for index in 0..pages {
            page_scratch.fill(index as u8 + 31);
            entries.push(LeafEntry {
                key: DirectoryKey {
                    section: Section::ColumnBlocks as u16,
                    flags: KEY_REQUIRED,
                    column: 0,
                    ordinal: index as u64,
                },
                page: writer
                    .append_stored(Codec::Raw, &page_scratch, PAGE as u64)
                    .unwrap(),
            });
        }
        let directory = writer.append_leaf(&entries, &mut node).unwrap();
        let summary = VolumeShape {
            layout: VolumeLayout::RowId,
            row_count: pages as u64,
            column_count: 1,
            group_count: pages as u64,
            rows: Some(RowBounds {
                min: 1,
                max: pages as i64,
            }),
            window: None,
        }
        .into_root(
            pages as u64,
            Some(DirectoryRoot {
                depth: 1,
                page: directory,
            }),
            None,
        )
        .unwrap();
        let finished = writer.finish(&summary).unwrap();
        file.flush().unwrap();
        drop(file);
        let (volume, open_counts) = measured(|| {
            FileBackedVolume::from_finished(
                &path,
                &finished,
                limits,
                &mut root_stored,
                &mut root_decoded,
            )
            .unwrap()
        });
        assert!(open_counts.1 < 64 * 1024, "open requested {open_counts:?}");
        if let Some(small_bytes) = previous_open_bytes {
            assert!(open_counts.1 <= small_bytes + 4096);
        }
        previous_open_bytes = Some(open_counts.1);
        // Active FD creation and all scratch reservations precede the read
        // meter. Every directory lookup and page read must reuse that storage.
        let lease = volume.lease().unwrap();
        let (_, read_counts) = measured(|| {
            let mut directory = lease.directory(&mut node, &mut []).unwrap();
            for entry in entries.iter().rev() {
                let found = directory.find(entry.key.into()).unwrap().unwrap();
                assert_eq!(found, *entry);
                let plan = lease.page(found.page).unwrap();
                let bytes = plan.read_into(&mut page_scratch, &mut []).unwrap();
                assert_eq!(bytes.len(), PAGE);
                assert!(
                    bytes
                        .iter()
                        .all(|byte| *byte == entry.key.ordinal as u8 + 31)
                );
            }
        });
        assert_eq!(read_counts, (0, 0));
        eprintln!(
            "V5 file bytes={} open_alloc={open_counts:?} page_read_alloc={read_counts:?}",
            finished.footer.file_length
        );
    }
}
