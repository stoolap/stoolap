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

//! Streaming directory construction from strictly key-ordered leaf entries.
//! Payload pages have already been written. The caller supplies the merge of
//! bounded metadata runs, a fixed DirectoryScratch and one encoding buffer.
//! No allocation, seek, payload ownership or volume-sized descriptor collection
//! is needed here. Scratch capacity must be charged for its complete lifetime.
//!
//! Full nodes are appended immediately. At completion partial nodes are carried
//! upward only as needed to meet previously emitted siblings at the same depth;
//! a sole root child is reused without writing a redundant parent. Each entry
//! reaches exactly one leaf, and all descriptors point backward in the file.

use std::fmt;
use std::io::Write;

use super::directory::{
    DirectoryError, DirectoryKey, DirectoryRoot, InteriorEntry, LeafEntry, Section,
    INTERIOR_ENTRY_BYTES, KEY_REQUIRED, MAX_DEPTH, MAX_FANOUT, NODE_HEADER_BYTES,
};
use super::envelope::{Codec, PageDescriptor};
use super::page_io::{PageIoError, PageWriter};

/// Maximum encoded node produced by this builder (fanout 64).
pub const ENCODING_BYTES: usize = NODE_HEADER_BYTES + MAX_FANOUT * INTERIOR_ENTRY_BYTES;
/// Depth includes leaves, hence 64^8 leaf entries fit exactly.
pub const MAX_ENTRIES: u64 = (MAX_FANOUT as u64).pow(MAX_DEPTH as u32);
const INTERIOR_LEVELS: usize = MAX_DEPTH as usize - 1;

const EMPTY_KEY: DirectoryKey = DirectoryKey {
    section: Section::RowIds as u16,
    flags: KEY_REQUIRED,
    column: u32::MAX,
    ordinal: 0,
};
const EMPTY_PAGE: PageDescriptor = PageDescriptor {
    offset: 0,
    stored_len: 1,
    decoded_len: 1,
    stored_checksum: 0,
    codec: Codec::Raw,
};
const EMPTY_LEAF: LeafEntry = LeafEntry {
    key: EMPTY_KEY,
    page: EMPTY_PAGE,
};
const EMPTY_INTERIOR: InteriorEntry = InteriorEntry {
    lower: EMPTY_KEY,
    upper: EMPTY_KEY,
    child: EMPTY_PAGE,
};

#[derive(Clone, Copy)]
struct Level {
    entries: [InteriorEntry; MAX_FANOUT],
    len: usize,
    count: u64,
}
impl Level {
    const EMPTY: Self = Self {
        entries: [EMPTY_INTERIOR; MAX_FANOUT],
        len: 0,
        count: 0,
    };
}

/// Fixed caller-owned capacity, reusable after the builder is dropped. Even an
/// empty volume does not allocate it internally or initialize all slots again.
pub struct DirectoryScratch {
    leaves: [LeafEntry; MAX_FANOUT],
    leaf_len: usize,
    levels: [Level; INTERIOR_LEVELS],
}
impl DirectoryScratch {
    pub const fn new() -> Self {
        Self {
            leaves: [EMPTY_LEAF; MAX_FANOUT],
            leaf_len: 0,
            levels: [Level::EMPTY; INTERIOR_LEVELS],
        }
    }
    fn reset(&mut self) {
        self.leaf_len = 0;
        for level in &mut self.levels {
            level.len = 0;
            level.count = 0;
        }
    }
}
impl Default for DirectoryScratch {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub enum DirectoryWriteError {
    Directory(DirectoryError),
    PageIo(PageIoError),
    ScratchTooSmall,
    PageLimitsTooSmall,
    Capacity,
    Poisoned,
    Finished,
}
impl fmt::Display for DirectoryWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V5 directory writer: {self:?}")
    }
}
impl std::error::Error for DirectoryWriteError {}
impl From<DirectoryError> for DirectoryWriteError {
    fn from(error: DirectoryError) -> Self {
        Self::Directory(error)
    }
}
impl From<PageIoError> for DirectoryWriteError {
    fn from(error: PageIoError) -> Self {
        Self::PageIo(error)
    }
}
type Result<T> = std::result::Result<T, DirectoryWriteError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BuiltDirectory {
    pub root: Option<DirectoryRoot>,
    pub entry_count: u64,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Open,
    Poisoned,
    Finished,
}

pub struct DirectoryWriter<'a, 'sink, W: Write + ?Sized> {
    writer: &'a mut PageWriter<'sink, W>,
    scratch: &'a mut DirectoryScratch,
    encoding: &'a mut [u8],
    previous: Option<DirectoryKey>,
    count: u64,
    complete_root: Option<DirectoryRoot>,
    state: State,
}
impl<'a, 'sink, W: Write + ?Sized> DirectoryWriter<'a, 'sink, W> {
    pub fn new(
        writer: &'a mut PageWriter<'sink, W>,
        scratch: &'a mut DirectoryScratch,
        encoding: &'a mut [u8],
    ) -> Result<Self> {
        writer.check_open()?;
        if encoding.len() < ENCODING_BYTES {
            return Err(DirectoryWriteError::ScratchTooSmall);
        }
        let limits = writer.limits();
        if limits.page_stored_bytes < ENCODING_BYTES as u64
            || limits.page_decoded_bytes < ENCODING_BYTES as u64
        {
            return Err(DirectoryWriteError::PageLimitsTooSmall);
        }
        scratch.reset();
        Ok(Self {
            writer,
            scratch,
            encoding,
            previous: None,
            count: 0,
            complete_root: None,
            state: State::Open,
        })
    }
    /// Invalid keys/references and exhaustion fail before accepting this entry.
    /// An error or unwind while emitting accepted entries makes the builder
    /// unusable; discard that unpublishable build instead of retrying a prefix.
    pub fn push(&mut self, entry: LeafEntry) -> Result<()> {
        self.check_open()?;
        if self.count == MAX_ENTRIES {
            return Err(DirectoryWriteError::Capacity);
        }
        entry.key.validate()?;
        if self
            .previous
            .is_some_and(|key| key.identity() >= entry.key.identity())
        {
            return Err(DirectoryError::KeyOrder.into());
        }
        self.writer.validate_reference(entry.page)?;
        self.state = State::Poisoned;
        self.scratch.leaves[self.scratch.leaf_len] = entry;
        self.scratch.leaf_len += 1;
        self.count += 1;
        self.previous = Some(entry.key);
        if self.scratch.leaf_len == MAX_FANOUT {
            self.flush_leaf()?;
        }
        self.state = State::Open;
        Ok(())
    }
    /// Return only the bounded root and exact entry count. The caller uses
    /// these in RootSummary and then finishes the enclosing PageWriter.
    pub fn finish(&mut self) -> Result<BuiltDirectory> {
        self.check_open()?;
        self.state = State::Poisoned;
        if self.scratch.leaf_len != 0 {
            self.flush_leaf()?;
        }
        let root = self.finish_levels()?;
        self.state = State::Finished;
        Ok(BuiltDirectory {
            root,
            entry_count: self.count,
        })
    }
    fn check_open(&self) -> Result<()> {
        match self.state {
            State::Open => Ok(()),
            State::Poisoned => Err(DirectoryWriteError::Poisoned),
            State::Finished => Err(DirectoryWriteError::Finished),
        }
    }
    fn flush_leaf(&mut self) -> Result<()> {
        let entries = &self.scratch.leaves[..self.scratch.leaf_len];
        let lower = entries[0].key;
        let upper = entries[entries.len() - 1].key;
        let count = entries.len() as u64;
        let child = self.writer.append_leaf(entries, self.encoding)?;
        self.scratch.leaf_len = 0;
        self.carry(
            1,
            InteriorEntry {
                lower,
                upper,
                child,
            },
            count,
        )
    }
    fn carry(&mut self, depth: u8, entry: InteriorEntry, count: u64) -> Result<()> {
        if depth == MAX_DEPTH {
            // Exactly 64^8 entries fill the last legal root. push rejects the
            // next entry before any state changes, so no depth9 carry exists.
            self.complete_root = Some(DirectoryRoot {
                depth,
                page: entry.child,
            });
            return Ok(());
        }
        let index = usize::from(depth - 1);
        let level = &mut self.scratch.levels[index];
        level.entries[level.len] = entry;
        level.len += 1;
        level.count += count; // Bounded by MAX_ENTRIES before every accepted push.
        if level.len == MAX_FANOUT {
            self.flush_level(index)?;
        }
        Ok(())
    }
    fn flush_level(&mut self, index: usize) -> Result<()> {
        let level = &self.scratch.levels[index];
        let lower = level.entries[0].lower;
        let upper = level.entries[level.len - 1].upper;
        let count = level.count;
        let depth = index as u8 + 2;
        let child = self.writer.append_interior(
            depth,
            count,
            &level.entries[..level.len],
            self.encoding,
        )?;
        self.scratch.levels[index].len = 0;
        self.scratch.levels[index].count = 0;
        self.carry(
            depth,
            InteriorEntry {
                lower,
                upper,
                child,
            },
            count,
        )
    }
    fn finish_levels(&mut self) -> Result<Option<DirectoryRoot>> {
        if self.complete_root.is_some() {
            return Ok(self.complete_root);
        }
        for index in 0..INTERIOR_LEVELS {
            let level = &self.scratch.levels[index];
            if level.len == 0 {
                continue;
            }
            if level.len == 1 && self.scratch.levels[index + 1..].iter().all(|l| l.len == 0) {
                return Ok(Some(DirectoryRoot {
                    depth: index as u8 + 1,
                    page: level.entries[0].child,
                }));
            }
            self.flush_level(index)?;
        }
        Ok(self.complete_root)
    }
}

#[cfg(test)]
mod tests {
    use super::super::directory::{DirectoryNode, Layout, NodeKind, RootSummary, RowBounds};
    use super::super::envelope::{FileIdentity, Footer, Header, ReadLimits};
    use super::super::page_io::{OpenedEnvelope, PageReadPlan, ReadAt};
    use super::*;
    use std::io;

    fn limits() -> ReadLimits {
        ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: ENCODING_BYTES as u64,
            page_decoded_bytes: ENCODING_BYTES as u64,
        }
    }
    fn header() -> Header {
        Header::new(FileIdentity::new(1, 1, 1).unwrap())
    }
    fn entry(ordinal: u64, page: PageDescriptor) -> LeafEntry {
        LeafEntry {
            key: DirectoryKey {
                ordinal,
                ..EMPTY_KEY
            },
            page,
        }
    }
    fn summary(directory: BuiltDirectory) -> RootSummary {
        RootSummary {
            layout: Layout::RowId,
            legacy_base: None,
            row_count: u64::from(directory.entry_count != 0),
            column_count: 1,
            group_count: u64::from(directory.entry_count != 0),
            entry_count: directory.entry_count,
            rows: (directory.entry_count != 0).then_some(RowBounds { min: 1, max: 1 }),
            window: None,
            directory: directory.root,
        }
    }
    struct Bytes<'a>(&'a [u8]);
    impl ReadAt for Bytes<'_> {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
            let start = usize::try_from(offset).unwrap();
            if start >= self.0.len() {
                return Ok(0);
            }
            let len = dst.len().min(self.0.len() - start);
            dst[..len].copy_from_slice(&self.0[start..start + len]);
            Ok(len)
        }
    }
    // Independent full traversal checks exact parent/child identities, depth,
    // CRCs, leaf order, entry sums and non-overlapping emitted directory nodes.
    fn walk(
        file: &[u8],
        page: PageDescriptor,
        footer: &Footer,
        next_key: &mut u64,
        locations: &mut Vec<(u64, u64)>,
    ) -> u64 {
        let start = page.offset as usize;
        let end = start + page.stored_len as usize;
        page.verify_stored_bytes(&file[start..end]).unwrap();
        let node = DirectoryNode::decode(&file[start..end], page, footer, &limits()).unwrap();
        locations.push((page.offset, end as u64));
        let mut count = 0;
        match node.kind() {
            NodeKind::Leaf => {
                for entry in node.leaves() {
                    let entry = entry.unwrap();
                    assert_eq!(entry.key.ordinal, *next_key);
                    *next_key += 1;
                    count += 1;
                }
            }
            NodeKind::Interior => {
                for (ordinal, child) in node.children().enumerate() {
                    let child = child.unwrap();
                    let bytes = &file[child.child.offset as usize
                        ..(child.child.offset + child.child.stored_len) as usize];
                    let child_node =
                        DirectoryNode::decode(bytes, child.child, footer, &limits()).unwrap();
                    node.validate_child(ordinal, &child_node).unwrap();
                    count += walk(file, child.child, footer, next_key, locations);
                }
            }
        }
        assert_eq!(count, node.subtree_entries());
        count
    }

    #[test]
    fn directory_streaming_boundaries_reopen_and_cover_every_key_once() {
        let mut scratch = DirectoryScratch::new();
        let mut encoding = [0; ENCODING_BYTES];
        for count in [0, 1, 63, 64, 65, 4095, 4096, 4097, 262144, 262145] {
            let mut file = Vec::new();
            let mut writer = PageWriter::new(&mut file, header(), limits()).unwrap();
            let payload = writer.append_stored(Codec::Raw, b"payload", 7).unwrap();
            let built = {
                let mut builder =
                    DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding).unwrap();
                for ordinal in 0..count {
                    builder.push(entry(ordinal, payload)).unwrap();
                }
                let result = builder.finish().unwrap();
                assert!(matches!(
                    builder.finish(),
                    Err(DirectoryWriteError::Finished)
                ));
                assert!(matches!(
                    builder.push(entry(count, payload)),
                    Err(DirectoryWriteError::Finished)
                ));
                result
            };
            assert_eq!(built.entry_count, count);
            let expected_depth = match count {
                0 => None,
                1..=64 => Some(1),
                65..=4096 => Some(2),
                4097..=262144 => Some(3),
                _ => Some(4),
            };
            assert_eq!(built.root.map(|root| root.depth), expected_depth);
            let finished = writer.finish(&summary(built)).unwrap();
            let opened = OpenedEnvelope::read(&Bytes(&file), file.len() as u64, &limits()).unwrap();
            opened.require_identity(header().identity).unwrap();
            assert_eq!(opened.footer, finished.footer);
            let mut root_bytes = [0; 128];
            let decoded = PageReadPlan::for_root(&opened.footer, &limits())
                .unwrap()
                .read_into(&Bytes(&file), &mut root_bytes, &mut [])
                .unwrap();
            let summary = RootSummary::decode(decoded, &opened.footer, &limits()).unwrap();
            let mut next_key = 0;
            let mut locations = Vec::new();
            if let Some(root) = built.root {
                let bytes = &file
                    [root.page.offset as usize..(root.page.offset + root.page.stored_len) as usize];
                let node =
                    DirectoryNode::decode(bytes, root.page, &opened.footer, &limits()).unwrap();
                summary.validate_directory_root(&node).unwrap();
                assert_eq!(
                    walk(
                        &file,
                        root.page,
                        &opened.footer,
                        &mut next_key,
                        &mut locations
                    ),
                    count
                );
            }
            assert_eq!(next_key, count);
            locations.sort_unstable();
            for pair in locations.windows(2) {
                assert!(pair[0].1 <= pair[1].0);
            }
        }
    }

    #[test]
    fn invalid_entry_preflight_preserves_buffered_tail_and_sink() {
        let mut file = Vec::new();
        let mut writer = PageWriter::new(&mut file, header(), limits()).unwrap();
        let payload = writer.append_stored(Codec::Raw, b"x", 1).unwrap();
        let mut scratch = DirectoryScratch::new();
        let mut encoding = [0; ENCODING_BYTES];
        let mut builder = DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding).unwrap();
        for i in 0..63 {
            builder.push(entry(i, payload)).unwrap();
        }
        let position = builder.writer.position();
        for bad in [
            entry(62, payload),
            LeafEntry {
                key: DirectoryKey {
                    flags: 0,
                    ..entry(63, payload).key
                },
                page: payload,
            },
            LeafEntry {
                key: DirectoryKey {
                    section: u16::MAX,
                    ..entry(63, payload).key
                },
                page: payload,
            },
            entry(
                63,
                PageDescriptor {
                    offset: position,
                    ..payload
                },
            ),
            entry(
                63,
                PageDescriptor {
                    offset: u64::MAX,
                    ..payload
                },
            ),
        ] {
            assert!(builder.push(bad).is_err());
            assert_eq!(builder.writer.position(), position);
            assert_eq!(builder.scratch.leaf_len, 63);
            assert_eq!(builder.count, 63);
        }
        builder.push(entry(63, payload)).unwrap();
        assert_eq!(builder.finish().unwrap().entry_count, 64);
    }

    #[test]
    fn exhausted_depth_rejects_the_next_entry_before_mutation() {
        assert_eq!(MAX_ENTRIES, 1 << 48);
        let mut file = Vec::new();
        let mut writer = PageWriter::new(&mut file, header(), limits()).unwrap();
        let page = writer.append_stored(Codec::Raw, b"x", 1).unwrap();
        let mut scratch = DirectoryScratch::new();
        let mut encoding = [0; ENCODING_BYTES];
        let mut builder = DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding).unwrap();
        // Exercise the terminal carry directly; building 2^48 entries would
        // hide this boundary behind petabytes of irrelevant fixture data.
        builder.count = MAX_ENTRIES;
        builder
            .carry(
                MAX_DEPTH,
                InteriorEntry {
                    child: page,
                    ..EMPTY_INTERIOR
                },
                MAX_ENTRIES,
            )
            .unwrap();
        let position = builder.writer.position();
        assert!(matches!(
            builder.push(entry(MAX_ENTRIES, page)),
            Err(DirectoryWriteError::Capacity)
        ));
        assert_eq!(builder.writer.position(), position);
        let result = builder.finish().unwrap();
        assert_eq!(
            result.root,
            Some(DirectoryRoot {
                depth: MAX_DEPTH,
                page
            })
        );
        assert_eq!(result.entry_count, MAX_ENTRIES);
    }

    #[test]
    fn scratch_limits_and_terminal_page_writer_fail_before_reset() {
        let mut file = Vec::new();
        let mut writer = PageWriter::new(&mut file, header(), limits()).unwrap();
        let mut scratch = DirectoryScratch::new();
        scratch.leaf_len = 17;
        let mut encoding = [0; ENCODING_BYTES];
        assert!(matches!(
            DirectoryWriter::new(
                &mut writer,
                &mut scratch,
                &mut encoding[..ENCODING_BYTES - 1]
            ),
            Err(DirectoryWriteError::ScratchTooSmall)
        ));
        assert_eq!(scratch.leaf_len, 17);
        writer
            .finish(&summary(BuiltDirectory {
                root: None,
                entry_count: 0,
            }))
            .unwrap();
        assert!(matches!(
            DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding),
            Err(DirectoryWriteError::PageIo(PageIoError::Finished))
        ));
        assert_eq!(scratch.leaf_len, 17);
        let mut file = Vec::new();
        let mut writer = PageWriter::new(
            &mut file,
            header(),
            ReadLimits {
                page_decoded_bytes: ENCODING_BYTES as u64 - 1,
                ..limits()
            },
        )
        .unwrap();
        assert!(matches!(
            DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding),
            Err(DirectoryWriteError::PageLimitsTooSmall)
        ));
        assert_eq!(scratch.leaf_len, 17);
    }

    struct FailingSink {
        remaining: usize,
        panic: bool,
    }
    impl Write for FailingSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.remaining == 0 {
                assert!(!self.panic, "injected sink panic after prefix");
                return Err(io::ErrorKind::Other.into());
            }
            let len = bytes.len().min(self.remaining);
            self.remaining -= len;
            Ok(len)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn emitted_prefix_error_and_caught_panic_poison_directory_builder() {
        for panic in [false, true] {
            let mut sink = FailingSink {
                remaining: 64 + 1 + 13,
                panic,
            };
            let mut writer = PageWriter::new(&mut sink, header(), limits()).unwrap();
            let page = writer.append_stored(Codec::Raw, b"x", 1).unwrap();
            let mut scratch = DirectoryScratch::new();
            let mut encoding = [0; ENCODING_BYTES];
            let mut builder =
                DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding).unwrap();
            for i in 0..63 {
                builder.push(entry(i, page)).unwrap();
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                builder.push(entry(63, page))
            }));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(result.unwrap().is_err());
            }
            assert!(builder.writer.is_poisoned());
            assert!(matches!(
                builder.push(entry(64, page)),
                Err(DirectoryWriteError::Poisoned)
            ));
            assert!(matches!(
                builder.finish(),
                Err(DirectoryWriteError::Poisoned)
            ));
        }
    }
}
