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

//! Bounded lazy directory lookup and explicit structural traversal. No File,
//! allocation or directory-sized collection is owned here. The caller supplies
//! an immutable ReadAt source, validates its manifest identity, and reserves
//! buffers before construction. Constructors perform no I/O.
//!
//! A lookup validates only the visited root-to-leaf path. A missing key is not
//! a certificate for the rest of the tree: payload readers must treat missing
//! required keys/sections as corruption according to their logical schema.
//! Ordinary cold open must not eagerly drain the diagnostic walker.
//!
//! A walker reads each directory page once, retaining one decoded buffer per
//! depth (at most eight) and a scalar frame stack. Initial node validation is
//! complete; subsequent entry advances decode only the next fixed-size entry.
//! Completion validates every child identity/range/depth, all advertised subtree
//! counts, root count, global key order and increasing physical postorder of
//! directory nodes. Payload extents/coverage and payload checksums are outside
//! this structural traversal. Unknown optional sections are yielded and counted.
//!
//! Stored scratch is reused between compressed nodes. Raw nodes read directly
//! into their decoded slot and need no separate stored buffer. The configured
//! stored-size bound is exposed for callers that need to handle arbitrary mixed
//! codecs; a smaller buffer (including empty for all-Raw files) fails before any
//! oversized compressed read. Root summary reads belong to PageReadPlan::for_root.
//!
//! Runtime errors are retained, and subsequent calls issue no I/O. Returning a
//! borrowed error preserves the original io::Error without allocating an Arc or
//! cloning its message. Callers must not interpret an earlier yielded prefix as
//! successful whole-tree validation when a later call reports corruption.

use std::fmt;

use super::directory::{
    DirectoryError, DirectoryKey, DirectoryNode, DirectoryRoot, InteriorEntry, LeafEntry, NodeKind,
    RootSummary, INTERIOR_ENTRY_BYTES, LEAF_ENTRY_BYTES, MAX_DEPTH, MAX_NODE_BYTES,
    NODE_HEADER_BYTES,
};
use super::envelope::{Codec, Footer, PageDescriptor, ReadLimits};
use super::page_io::{PageIoError, PageReadPlan, ReadAt};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct KeyIdentity {
    pub section: u16,
    pub column: u32,
    pub ordinal: u64,
}
impl From<DirectoryKey> for KeyIdentity {
    fn from(key: DirectoryKey) -> Self {
        Self {
            section: key.section,
            column: key.column,
            ordinal: key.ordinal,
        }
    }
}

#[derive(Debug)]
pub enum DirectoryReadError {
    Directory(DirectoryError),
    PageIo(PageIoError),
    BufferTooSmall,
    EntryCount,
    KeyOrder,
    PhysicalOrder,
    InterruptedOperation,
}
impl From<DirectoryError> for DirectoryReadError {
    fn from(error: DirectoryError) -> Self {
        Self::Directory(error)
    }
}
impl From<PageIoError> for DirectoryReadError {
    fn from(error: PageIoError) -> Self {
        Self::PageIo(error)
    }
}
impl fmt::Display for DirectoryReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Directory(error) => write!(f, "V5 directory read: {error}"),
            Self::PageIo(error) => write!(f, "V5 directory read: {error}"),
            other => write!(f, "invalid V5 directory read: {other:?}"),
        }
    }
}
impl std::error::Error for DirectoryReadError {}
type Result<T> = std::result::Result<T, DirectoryReadError>;
type BorrowedResult<'a, T> = std::result::Result<T, &'a DirectoryReadError>;

/// Validate metadata and query requirements before reserving or allocating.
/// The stored bound is sufficient for all nodes allowed by limits; it is not
/// an assertion that the file actually contains a compressed node of that size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectoryBufferRequirements {
    pub lookup_node_bytes: usize,
    pub walker_slots: usize,
    pub walker_node_bytes: usize,
    pub compressed_stored_bound: usize,
    pub first_node_stored_bytes: usize,
}
impl DirectoryBufferRequirements {
    /// Checked combined capacity sufficient for lookup with any allowed codec.
    /// All-Raw callers may omit stored scratch and reserve lookup_node_bytes.
    pub fn max_lookup_bytes(&self) -> Result<usize> {
        self.lookup_node_bytes
            .checked_add(self.compressed_stored_bound)
            .ok_or(PageIoError::AddressSpace.into())
    }
    /// Checked combined capacity sufficient for a full structural traversal.
    pub fn max_walker_bytes(&self) -> Result<usize> {
        self.walker_node_bytes
            .checked_add(self.compressed_stored_bound)
            .ok_or(PageIoError::AddressSpace.into())
    }
    pub fn new(footer: &Footer, summary: &RootSummary, limits: &ReadLimits) -> Result<Self> {
        summary.validate()?;
        PageReadPlan::for_root(footer, limits)?;
        let Some(root) = summary.directory else {
            return Ok(Self {
                lookup_node_bytes: 0,
                walker_slots: 0,
                walker_node_bytes: 0,
                compressed_stored_bound: 0,
                first_node_stored_bytes: 0,
            });
        };
        let plan = PageReadPlan::for_directory(footer, root.page, limits)?;
        let compressed_stored_bound =
            usize::try_from(limits.page_stored_bytes).map_err(|_| PageIoError::AddressSpace)?;
        let slots = usize::from(root.depth);
        Ok(Self {
            lookup_node_bytes: MAX_NODE_BYTES,
            walker_slots: slots,
            walker_node_bytes: slots * MAX_NODE_BYTES,
            compressed_stored_bound,
            first_node_stored_bytes: if root.page.codec == Codec::Raw {
                0
            } else {
                plan.stored_buffer_len()
            },
        })
    }
}

struct Context<'a, R: ReadAt + ?Sized> {
    source: &'a R,
    footer: Footer,
    summary: RootSummary,
    limits: ReadLimits,
}
#[derive(Clone, Copy)]
enum ExpectedNode {
    Root(DirectoryRoot, u64),
    Child {
        parent_depth: u8,
        entry: InteriorEntry,
    },
}
#[derive(Clone, Copy)]
struct Frame {
    page: PageDescriptor,
    kind: NodeKind,
    depth: u8,
    count: u16,
    next: u16,
    advertised: u64,
    observed: u64,
}
impl Frame {
    fn from_node(node: DirectoryNode<'_>, page: PageDescriptor) -> Self {
        Self {
            page,
            kind: node.kind(),
            depth: node.depth(),
            count: node.len() as u16,
            next: 0,
            advertised: node.subtree_entries(),
            observed: 0,
        }
    }
    fn leaf(&self, buffer: &[u8], ordinal: usize) -> Result<LeafEntry> {
        LeafEntry::decode(self.entry_bytes(buffer, ordinal, LEAF_ENTRY_BYTES)?).map_err(Into::into)
    }
    fn child(&self, buffer: &[u8], ordinal: usize) -> Result<InteriorEntry> {
        InteriorEntry::decode(self.entry_bytes(buffer, ordinal, INTERIOR_ENTRY_BYTES)?)
            .map_err(Into::into)
    }
    fn entry_bytes<'a>(&self, buffer: &'a [u8], ordinal: usize, size: usize) -> Result<&'a [u8]> {
        if ordinal >= usize::from(self.count) {
            return Err(DirectoryError::Length.into());
        }
        let start = NODE_HEADER_BYTES + ordinal * size;
        buffer
            .get(start..start + size)
            .ok_or(DirectoryError::Length.into())
    }
}
fn read_node<R: ReadAt + ?Sized>(
    context: &Context<'_, R>,
    page: PageDescriptor,
    expected: ExpectedNode,
    node_buffer: &mut [u8],
    stored: &mut [u8],
) -> Result<Frame> {
    let plan = PageReadPlan::for_directory(&context.footer, page, &context.limits)?;
    let bytes = match page.codec {
        Codec::Raw => plan.read_into(context.source, node_buffer, &mut [])?,
        Codec::Lz4Block => plan.read_into(context.source, stored, node_buffer)?,
    };
    let node = DirectoryNode::decode(bytes, page, &context.footer, &context.limits)?;
    match expected {
        ExpectedNode::Root(root, count) => {
            if root.page != page || root.depth != node.depth() || count != node.subtree_entries() {
                return Err(DirectoryError::ChildMismatch.into());
            }
        }
        ExpectedNode::Child {
            parent_depth,
            entry,
        } => {
            let (lower, upper) = node.key_range();
            if entry.child != page
                || node.depth() + 1 != parent_depth
                || lower.identity() != entry.lower.identity()
                || upper.identity() != entry.upper.identity()
            {
                return Err(DirectoryError::ChildMismatch.into());
            }
        }
    }
    Ok(Frame::from_node(node, page))
}

pub struct DirectoryLookup<'a, R: ReadAt + ?Sized> {
    context: Context<'a, R>,
    node: &'a mut [u8],
    stored: &'a mut [u8],
    error: Option<DirectoryReadError>,
}
impl<'a, R: ReadAt + ?Sized> DirectoryLookup<'a, R> {
    pub fn new(
        source: &'a R,
        footer: Footer,
        summary: RootSummary,
        limits: ReadLimits,
        node: &'a mut [u8],
        stored: &'a mut [u8],
    ) -> Result<Self> {
        let requirements = DirectoryBufferRequirements::new(&footer, &summary, &limits)?;
        if node.len() < requirements.lookup_node_bytes
            || stored.len() < requirements.first_node_stored_bytes
        {
            return Err(DirectoryReadError::BufferTooSmall);
        }
        Ok(Self {
            context: Context {
                source,
                footer,
                summary,
                limits,
            },
            node,
            stored,
            error: None,
        })
    }
    pub fn error(&self) -> Option<&DirectoryReadError> {
        self.error.as_ref()
    }
    pub fn find(&mut self, key: KeyIdentity) -> BorrowedResult<'_, Option<LeafEntry>> {
        if self.error.is_none() {
            // ReadAt is supplied by the caller and may unwind. Keep a scalar
            // poison sentinel until the entire operation completes normally.
            self.error = Some(DirectoryReadError::InterruptedOperation);
            match self.find_inner(key) {
                Ok(found) => {
                    self.error = None;
                    return Ok(found);
                }
                Err(error) => self.error = Some(error),
            }
        }
        Err(self.error.as_ref().expect("retained directory error"))
    }
    fn find_inner(&mut self, key: KeyIdentity) -> Result<Option<LeafEntry>> {
        let Some(root) = self.context.summary.directory else {
            return Ok(None);
        };
        let mut page = root.page;
        let mut expected = ExpectedNode::Root(root, self.context.summary.entry_count);
        loop {
            let frame = read_node(&self.context, page, expected, self.node, self.stored)?;
            let mut child = None;
            for ordinal in 0..usize::from(frame.count) {
                match frame.kind {
                    NodeKind::Leaf => {
                        let entry = frame.leaf(self.node, ordinal)?;
                        match KeyIdentity::from(entry.key).cmp(&key) {
                            std::cmp::Ordering::Less => (),
                            std::cmp::Ordering::Equal => return Ok(Some(entry)),
                            std::cmp::Ordering::Greater => return Ok(None),
                        }
                    }
                    NodeKind::Interior => {
                        let entry = frame.child(self.node, ordinal)?;
                        if key < KeyIdentity::from(entry.lower) {
                            return Ok(None);
                        }
                        if key <= KeyIdentity::from(entry.upper) {
                            child = Some(entry);
                            break;
                        }
                    }
                }
            }
            let Some(entry) = child else {
                return Ok(None);
            };
            page = entry.child;
            expected = ExpectedNode::Child {
                parent_depth: frame.depth,
                entry,
            };
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WalkSummary {
    pub entry_count: u64,
    pub node_count: u64,
    pub depth: u8,
}
pub struct DirectoryWalker<'a, R: ReadAt + ?Sized> {
    context: Context<'a, R>,
    nodes: &'a mut [[u8; MAX_NODE_BYTES]],
    stored: &'a mut [u8],
    frames: [Option<Frame>; MAX_DEPTH as usize],
    frame_count: usize,
    previous_key: Option<KeyIdentity>,
    previous_completed_end: Option<u64>,
    summary: WalkSummary,
    started: bool,
    finished: bool,
    error: Option<DirectoryReadError>,
}
impl<'a, R: ReadAt + ?Sized> DirectoryWalker<'a, R> {
    pub fn new(
        source: &'a R,
        footer: Footer,
        summary: RootSummary,
        limits: ReadLimits,
        nodes: &'a mut [[u8; MAX_NODE_BYTES]],
        stored: &'a mut [u8],
    ) -> Result<Self> {
        let requirements = DirectoryBufferRequirements::new(&footer, &summary, &limits)?;
        if nodes.len() < requirements.walker_slots
            || stored.len() < requirements.first_node_stored_bytes
        {
            return Err(DirectoryReadError::BufferTooSmall);
        }
        Ok(Self {
            context: Context {
                source,
                footer,
                summary,
                limits,
            },
            nodes,
            stored,
            frames: [None; MAX_DEPTH as usize],
            frame_count: 0,
            previous_key: None,
            previous_completed_end: None,
            summary: WalkSummary::default(),
            started: false,
            finished: false,
            error: None,
        })
    }
    pub fn error(&self) -> Option<&DirectoryReadError> {
        self.error.as_ref()
    }
    pub fn validated_summary(&self) -> Option<WalkSummary> {
        (self.finished && self.error.is_none()).then_some(self.summary)
    }
    pub fn next_entry(&mut self) -> BorrowedResult<'_, Option<LeafEntry>> {
        if self.error.is_none() {
            self.error = Some(DirectoryReadError::InterruptedOperation);
            match self.advance() {
                Ok(entry) => {
                    self.error = None;
                    return Ok(entry);
                }
                Err(error) => self.error = Some(error),
            }
        }
        Err(self.error.as_ref().expect("retained directory error"))
    }
    /// Explicitly drain remaining entries. This is a diagnostic/finish validator,
    /// not an eager cold-open prerequisite. Success certifies directory structure
    /// only; payload interpretation, presence and coverage remain caller checks.
    pub fn finish_validation(&mut self) -> BorrowedResult<'_, WalkSummary> {
        if self.error.is_none() {
            self.error = Some(DirectoryReadError::InterruptedOperation);
            loop {
                match self.advance() {
                    Ok(Some(_)) => (),
                    Ok(None) => {
                        self.error = None;
                        return Ok(self.summary);
                    }
                    Err(error) => {
                        self.error = Some(error);
                        break;
                    }
                }
            }
        }
        Err(self.error.as_ref().expect("retained directory error"))
    }
    fn push_node(&mut self, page: PageDescriptor, expected: ExpectedNode) -> Result<()> {
        if self.frame_count == MAX_DEPTH as usize {
            return Err(DirectoryError::Depth.into());
        }
        if self
            .previous_completed_end
            .is_some_and(|end| page.offset < end)
        {
            return Err(DirectoryReadError::PhysicalOrder);
        }
        let frame = read_node(
            &self.context,
            page,
            expected,
            &mut self.nodes[self.frame_count],
            self.stored,
        )?;
        self.frames[self.frame_count] = Some(frame);
        self.frame_count += 1;
        self.summary.node_count = self
            .summary
            .node_count
            .checked_add(1)
            .ok_or(DirectoryReadError::EntryCount)?;
        Ok(())
    }
    fn advance(&mut self) -> Result<Option<LeafEntry>> {
        if self.finished {
            return Ok(None);
        }
        if !self.started {
            self.started = true;
            if let Some(root) = self.context.summary.directory {
                self.summary.depth = root.depth;
                self.push_node(
                    root.page,
                    ExpectedNode::Root(root, self.context.summary.entry_count),
                )?;
            }
        }
        while self.frame_count != 0 {
            let index = self.frame_count - 1;
            let mut frame = self.frames[index].expect("active directory frame");
            if frame.next == frame.count {
                if frame.observed != frame.advertised {
                    return Err(DirectoryReadError::EntryCount);
                }
                if self
                    .previous_completed_end
                    .is_some_and(|end| frame.page.offset < end)
                {
                    return Err(DirectoryReadError::PhysicalOrder);
                }
                self.previous_completed_end = Some(
                    frame
                        .page
                        .offset
                        .checked_add(frame.page.stored_len)
                        .ok_or(DirectoryError::PhysicalOrder)?,
                );
                self.frames[index] = None;
                self.frame_count -= 1;
                if self.frame_count != 0 {
                    let parent = self.frames[self.frame_count - 1]
                        .as_mut()
                        .expect("active parent frame");
                    parent.observed = parent
                        .observed
                        .checked_add(frame.observed)
                        .ok_or(DirectoryReadError::EntryCount)?;
                    if parent.observed > parent.advertised {
                        return Err(DirectoryReadError::EntryCount);
                    }
                }
                continue;
            }
            let ordinal = usize::from(frame.next);
            frame.next += 1;
            match frame.kind {
                NodeKind::Leaf => {
                    let entry = frame.leaf(&self.nodes[index], ordinal)?;
                    let key = KeyIdentity::from(entry.key);
                    if self.previous_key.is_some_and(|previous| previous >= key) {
                        return Err(DirectoryReadError::KeyOrder);
                    }
                    self.previous_key = Some(key);
                    frame.observed = frame
                        .observed
                        .checked_add(1)
                        .ok_or(DirectoryReadError::EntryCount)?;
                    self.summary.entry_count = self
                        .summary
                        .entry_count
                        .checked_add(1)
                        .ok_or(DirectoryReadError::EntryCount)?;
                    self.frames[index] = Some(frame);
                    return Ok(Some(entry));
                }
                NodeKind::Interior => {
                    let entry = frame.child(&self.nodes[index], ordinal)?;
                    self.frames[index] = Some(frame);
                    self.push_node(
                        entry.child,
                        ExpectedNode::Child {
                            parent_depth: frame.depth,
                            entry,
                        },
                    )?;
                }
            }
        }
        if self.summary.entry_count != self.context.summary.entry_count {
            return Err(DirectoryReadError::EntryCount);
        }
        self.finished = true;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::super::directory::{
        encode_interior, encode_leaf, Layout, RowBounds, Section, KEY_OPTIONAL, KEY_REQUIRED,
    };
    use super::super::directory_writer::{DirectoryScratch, DirectoryWriter, ENCODING_BYTES};
    use super::super::envelope::{FileIdentity, Header};
    use super::super::page_io::PageWriter;
    use super::*;
    use std::cell::Cell;
    use std::io;

    fn limits() -> ReadLimits {
        ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: 8192,
            page_decoded_bytes: 8192,
        }
    }
    fn header() -> Header {
        Header::new(FileIdentity::new(1, 1, 1).unwrap())
    }
    fn key(ordinal: u64) -> DirectoryKey {
        DirectoryKey {
            section: Section::RowIds as u16,
            flags: KEY_REQUIRED,
            column: u32::MAX,
            ordinal,
        }
    }
    fn lookup_key(ordinal: u64) -> KeyIdentity {
        key(ordinal).into()
    }
    struct Image {
        bytes: Vec<u8>,
        footer: Footer,
        summary: RootSummary,
    }
    fn summary(root: Option<DirectoryRoot>, count: u64) -> RootSummary {
        RootSummary {
            layout: Layout::RowId,
            legacy_base: None,
            row_count: u64::from(count != 0),
            column_count: 0,
            group_count: u64::from(count != 0),
            entry_count: count,
            rows: (count != 0).then_some(RowBounds { min: 0, max: 0 }),
            window: None,
            directory: root,
        }
    }
    fn build(count: u64) -> Image {
        let mut bytes = Vec::new();
        let mut writer = PageWriter::new(&mut bytes, header(), limits()).unwrap();
        let payload = writer.append_stored(Codec::Raw, b"x", 1).unwrap();
        let mut scratch = DirectoryScratch::new();
        let mut encoding = [0; ENCODING_BYTES];
        let mut builder = DirectoryWriter::new(&mut writer, &mut scratch, &mut encoding).unwrap();
        for ordinal in 0..count {
            builder
                .push(LeafEntry {
                    key: key(ordinal),
                    page: payload,
                })
                .unwrap();
        }
        let built = builder.finish().unwrap();
        let summary = summary(built.root, built.entry_count);
        let finished = writer.finish(&summary).unwrap();
        Image {
            bytes,
            footer: finished.footer,
            summary,
        }
    }
    struct Source<'a> {
        bytes: &'a [u8],
        calls: Cell<usize>,
        fail_call: usize,
        panic_call: usize,
    }
    impl<'a> Source<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self {
                bytes,
                calls: Cell::new(0),
                fail_call: 0,
                panic_call: 0,
            }
        }
    }
    impl ReadAt for Source<'_> {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
            let call = self.calls.get() + 1;
            self.calls.set(call);
            assert_ne!(call, self.panic_call, "injected positioned-read panic");
            if call == self.fail_call {
                return Err(io::ErrorKind::PermissionDenied.into());
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
    fn append(bytes: &mut Vec<u8>, data: &[u8], codec: Codec, decoded_len: u64) -> PageDescriptor {
        let page = PageDescriptor {
            offset: bytes.len() as u64,
            stored_len: data.len() as u64,
            decoded_len,
            stored_checksum: crc32fast::hash(data),
            codec,
        };
        bytes.extend_from_slice(data);
        page
    }
    fn leaf(bytes: &mut Vec<u8>, key: DirectoryKey, payload: PageDescriptor) -> PageDescriptor {
        let mut data = [0; 80];
        encode_leaf(&[LeafEntry { key, page: payload }], &mut data).unwrap();
        append(bytes, &data, Codec::Raw, 80)
    }
    fn parent(
        bytes: &mut Vec<u8>,
        depth: u8,
        count: u64,
        entries: &[InteriorEntry],
    ) -> PageDescriptor {
        let mut data = [0; ENCODING_BYTES];
        let len = encode_interior(depth, count, entries, &mut data).unwrap();
        append(bytes, &data[..len], Codec::Raw, len as u64)
    }
    fn seal(mut bytes: Vec<u8>, root: DirectoryRoot, count: u64) -> Image {
        let summary = summary(Some(root), count);
        let data = summary.encode().unwrap();
        let root_page = append(&mut bytes, &data, Codec::Raw, 128);
        let footer = Footer {
            file_length: bytes.len() as u64 + 64,
            root: root_page,
        };
        bytes.extend_from_slice(&footer.encode().unwrap());
        Image {
            bytes,
            footer,
            summary,
        }
    }
    fn start_image() -> (Vec<u8>, PageDescriptor) {
        let mut bytes = header().encode().unwrap().to_vec();
        let payload = append(&mut bytes, b"x", Codec::Raw, 1);
        (bytes, payload)
    }

    #[test]
    fn directory_lookup_reads_only_the_requested_path() {
        let image = build(10_000);
        let source = Source::new(&image.bytes);
        let requirements =
            DirectoryBufferRequirements::new(&image.footer, &image.summary, &limits()).unwrap();
        assert_eq!(requirements.walker_slots, 3);
        assert_eq!(requirements.lookup_node_bytes, MAX_NODE_BYTES);
        assert_eq!(requirements.first_node_stored_bytes, 0);
        let mut node = [0; MAX_NODE_BYTES];
        let mut lookup = DirectoryLookup::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut node,
            &mut [],
        )
        .unwrap();
        assert_eq!(source.calls.get(), 0);
        for ordinal in [0, 63, 64, 4095, 4096, 9999] {
            let before = source.calls.get();
            assert_eq!(
                lookup
                    .find(lookup_key(ordinal))
                    .unwrap()
                    .unwrap()
                    .key
                    .ordinal,
                ordinal
            );
            assert_eq!(source.calls.get() - before, 3);
        }
        let before = source.calls.get();
        assert!(lookup.find(lookup_key(10_000)).unwrap().is_none());
        assert_eq!(source.calls.get() - before, 1);
        assert!(lookup.error().is_none());
    }

    #[test]
    fn directory_walk_reads_each_node_once_and_validates_only_at_completion() {
        for count in [0u64, 1, 64, 65, 4096, 4097, 10_000] {
            let image = build(count);
            let source = Source::new(&image.bytes);
            let requirements =
                DirectoryBufferRequirements::new(&image.footer, &image.summary, &limits()).unwrap();
            let mut slots = vec![[0; MAX_NODE_BYTES]; requirements.walker_slots];
            let mut walker = DirectoryWalker::new(
                &source,
                image.footer,
                image.summary,
                limits(),
                &mut slots,
                &mut [],
            )
            .unwrap();
            assert_eq!(source.calls.get(), 0);
            assert!(walker.validated_summary().is_none());
            for expected in 0..count {
                assert_eq!(walker.next_entry().unwrap().unwrap().key.ordinal, expected);
                assert!(walker.validated_summary().is_none());
            }
            assert!(walker.next_entry().unwrap().is_none());
            let validated = walker.validated_summary().unwrap();
            assert_eq!(validated.entry_count, count);
            let mut at_level = count.div_ceil(64);
            let mut expected_nodes = at_level;
            while at_level > 1 {
                at_level = at_level.div_ceil(64);
                expected_nodes += at_level;
            }
            assert_eq!(validated.node_count, expected_nodes);
            assert_eq!(source.calls.get() as u64, expected_nodes);
            assert_eq!(walker.finish_validation().unwrap(), validated);
            assert!(walker.next_entry().unwrap().is_none());
            assert_eq!(source.calls.get() as u64, expected_nodes);
        }
        assert!(std::mem::size_of::<[Option<Frame>; MAX_DEPTH as usize]>() <= 1024);
    }

    #[test]
    fn directory_count_corruption_is_not_hidden_by_valid_lookup_path() {
        let (mut bytes, payload) = start_image();
        let left = leaf(&mut bytes, key(0), payload);
        let right = leaf(&mut bytes, key(1), payload);
        let root = parent(
            &mut bytes,
            2,
            3,
            &[
                InteriorEntry {
                    lower: key(0),
                    upper: key(0),
                    child: left,
                },
                InteriorEntry {
                    lower: key(1),
                    upper: key(1),
                    child: right,
                },
            ],
        );
        let image = seal(
            bytes,
            DirectoryRoot {
                depth: 2,
                page: root,
            },
            3,
        );
        let source = Source::new(&image.bytes);
        let mut node = [0; MAX_NODE_BYTES];
        let mut lookup = DirectoryLookup::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut node,
            &mut [],
        )
        .unwrap();
        assert!(lookup.find(lookup_key(0)).unwrap().is_some());
        let mut nodes = [[0; MAX_NODE_BYTES]; 2];
        let mut walker = DirectoryWalker::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut nodes,
            &mut [],
        )
        .unwrap();
        assert_eq!(walker.next_entry().unwrap().unwrap().key.ordinal, 0);
        assert_eq!(walker.next_entry().unwrap().unwrap().key.ordinal, 1);
        assert!(matches!(
            walker.next_entry(),
            Err(DirectoryReadError::EntryCount)
        ));
        assert!(walker.validated_summary().is_none());
        let calls = source.calls.get();
        let error = walker.error().unwrap() as *const DirectoryReadError;
        assert!(std::ptr::eq(walker.finish_validation().unwrap_err(), error));
        assert_eq!(source.calls.get(), calls);
    }

    #[test]
    fn directory_cross_subtree_physical_reorder_fails_streaming_validation() {
        // Child-root descriptors themselves ascend, but the right subtree's
        // leaf was physically written before the left subtree. This passes
        // local node checks and visited-path lookup; full postorder rejects it.
        let (mut bytes, payload) = start_image();
        let right_leaf = leaf(&mut bytes, key(1), payload);
        let left_leaf = leaf(&mut bytes, key(0), payload);
        let left = parent(
            &mut bytes,
            2,
            1,
            &[InteriorEntry {
                lower: key(0),
                upper: key(0),
                child: left_leaf,
            }],
        );
        let right = parent(
            &mut bytes,
            2,
            1,
            &[InteriorEntry {
                lower: key(1),
                upper: key(1),
                child: right_leaf,
            }],
        );
        let root = parent(
            &mut bytes,
            3,
            2,
            &[
                InteriorEntry {
                    lower: key(0),
                    upper: key(0),
                    child: left,
                },
                InteriorEntry {
                    lower: key(1),
                    upper: key(1),
                    child: right,
                },
            ],
        );
        let image = seal(
            bytes,
            DirectoryRoot {
                depth: 3,
                page: root,
            },
            2,
        );
        let source = Source::new(&image.bytes);
        let mut node = [0; MAX_NODE_BYTES];
        let mut lookup = DirectoryLookup::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut node,
            &mut [],
        )
        .unwrap();
        assert_eq!(lookup.find(lookup_key(1)).unwrap().unwrap().key.ordinal, 1);
        let before = source.calls.get();
        let mut nodes = [[0; MAX_NODE_BYTES]; 3];
        let mut walker = DirectoryWalker::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut nodes,
            &mut [],
        )
        .unwrap();
        assert_eq!(walker.next_entry().unwrap().unwrap().key.ordinal, 0);
        assert!(matches!(
            walker.next_entry(),
            Err(DirectoryReadError::PhysicalOrder)
        ));
        assert_eq!(source.calls.get() - before, 4); // Reject the reordered leaf before its I/O.
        assert!(walker.validated_summary().is_none());
    }

    #[test]
    fn directory_child_range_depth_and_late_crc_errors_are_sticky() {
        for failure in 0..3 {
            let (mut bytes, payload) = start_image();
            let left = leaf(&mut bytes, key(0), payload);
            let mut right = leaf(&mut bytes, key(if failure == 0 { 2 } else { 1 }), payload);
            if failure == 2 {
                right.stored_checksum ^= 1;
            }
            let depth = if failure == 1 { 3 } else { 2 };
            let root = parent(
                &mut bytes,
                depth,
                2,
                &[
                    InteriorEntry {
                        lower: key(0),
                        upper: key(0),
                        child: left,
                    },
                    InteriorEntry {
                        lower: key(1),
                        upper: key(1),
                        child: right,
                    },
                ],
            );
            let image = seal(bytes, DirectoryRoot { depth, page: root }, 2);
            let source = Source::new(&image.bytes);
            let mut node = [0; MAX_NODE_BYTES];
            let mut lookup = DirectoryLookup::new(
                &source,
                image.footer,
                image.summary,
                limits(),
                &mut node,
                &mut [],
            )
            .unwrap();
            if failure != 1 {
                assert!(lookup.find(lookup_key(0)).unwrap().is_some());
            }
            assert!(lookup.find(lookup_key(1)).is_err());
            let calls = source.calls.get();
            let first = lookup.error().unwrap() as *const DirectoryReadError;
            assert!(std::ptr::eq(lookup.find(lookup_key(0)).unwrap_err(), first));
            assert_eq!(source.calls.get(), calls);
            let mut nodes = [[0; MAX_NODE_BYTES]; 3];
            let mut walker = DirectoryWalker::new(
                &source,
                image.footer,
                image.summary,
                limits(),
                &mut nodes,
                &mut [],
            )
            .unwrap();
            if failure != 1 {
                assert_eq!(walker.next_entry().unwrap().unwrap().key.ordinal, 0);
            }
            assert!(walker.finish_validation().is_err());
            assert!(walker.validated_summary().is_none());
            let calls = source.calls.get();
            assert!(walker.next_entry().is_err());
            assert_eq!(source.calls.get(), calls);
        }
    }

    #[test]
    fn directory_unknown_optional_keys_are_yielded_and_counted() {
        let (mut bytes, payload) = start_image();
        let unknown = DirectoryKey {
            section: 65000,
            flags: KEY_OPTIONAL,
            column: 7,
            ordinal: 99,
        };
        let root = leaf(&mut bytes, unknown, payload);
        let image = seal(
            bytes,
            DirectoryRoot {
                depth: 1,
                page: root,
            },
            1,
        );
        let source = Source::new(&image.bytes);
        let mut node = [0; MAX_NODE_BYTES];
        let mut lookup = DirectoryLookup::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut node,
            &mut [],
        )
        .unwrap();
        assert_eq!(lookup.find(unknown.into()).unwrap().unwrap().key, unknown);
        let mut nodes = [[0; MAX_NODE_BYTES]; 1];
        let mut walker = DirectoryWalker::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut nodes,
            &mut [],
        )
        .unwrap();
        let entry = walker.next_entry().unwrap().unwrap();
        assert!(entry.key.skip_unknown_optional());
        assert_eq!(walker.finish_validation().unwrap().entry_count, 1);
    }

    #[test]
    fn directory_compressed_nodes_use_exact_shared_scratch() {
        let (mut bytes, payload) = start_image();
        let mut raw = [0; 80];
        encode_leaf(
            &[LeafEntry {
                key: key(0),
                page: payload,
            }],
            &mut raw,
        )
        .unwrap();
        let compressed = lz4_flex::block::compress(&raw);
        let root = append(&mut bytes, &compressed, Codec::Lz4Block, 80);
        let image = seal(
            bytes,
            DirectoryRoot {
                depth: 1,
                page: root,
            },
            1,
        );
        let source = Source::new(&image.bytes);
        let requirements =
            DirectoryBufferRequirements::new(&image.footer, &image.summary, &limits()).unwrap();
        assert_eq!(requirements.first_node_stored_bytes, compressed.len());
        let mut node = [0xa5; MAX_NODE_BYTES];
        assert!(matches!(
            DirectoryLookup::new(
                &source,
                image.footer,
                image.summary,
                limits(),
                &mut node,
                &mut []
            ),
            Err(DirectoryReadError::BufferTooSmall)
        ));
        assert_eq!(source.calls.get(), 0);
        let mut stored = vec![0; compressed.len()];
        let mut lookup = DirectoryLookup::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut node,
            &mut stored,
        )
        .unwrap();
        assert!(lookup.find(lookup_key(0)).unwrap().is_some());
        assert!(node[80..].iter().all(|&b| b == 0xa5));
        let mut nodes = [[0xa5; MAX_NODE_BYTES]; 1];
        let mut walker = DirectoryWalker::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut nodes,
            &mut stored,
        )
        .unwrap();
        assert_eq!(walker.finish_validation().unwrap().node_count, 1);
        assert!(nodes[0][80..].iter().all(|&b| b == 0xa5));
    }

    #[test]
    fn directory_invalid_metadata_and_buffers_fail_without_io() {
        let image = build(65);
        let source = Source::new(&image.bytes);
        let mut node = [0; MAX_NODE_BYTES];
        assert!(DirectoryLookup::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut node[..MAX_NODE_BYTES - 1],
            &mut []
        )
        .is_err());
        let mut nodes = [[0; MAX_NODE_BYTES]; 1];
        assert!(DirectoryWalker::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut nodes,
            &mut []
        )
        .is_err());
        let bad = RootSummary {
            entry_count: 0,
            ..image.summary
        };
        assert!(DirectoryBufferRequirements::new(&image.footer, &bad, &limits()).is_err());
        let requirements =
            DirectoryBufferRequirements::new(&image.footer, &image.summary, &limits()).unwrap();
        assert_eq!(
            requirements.max_lookup_bytes().unwrap(),
            MAX_NODE_BYTES + 8192
        );
        assert_eq!(
            requirements.max_walker_bytes().unwrap(),
            MAX_NODE_BYTES * 2 + 8192
        );
        let overflow = DirectoryBufferRequirements {
            compressed_stored_bound: usize::MAX,
            ..requirements
        };
        assert!(overflow.max_lookup_bytes().is_err());
        assert!(overflow.max_walker_bytes().is_err());
        assert_eq!(source.calls.get(), 0);
        let empty = build(0);
        let empty_source = Source::new(&empty.bytes);
        let mut lookup = DirectoryLookup::new(
            &empty_source,
            empty.footer,
            empty.summary,
            limits(),
            &mut [],
            &mut [],
        )
        .unwrap();
        assert!(lookup.find(lookup_key(0)).unwrap().is_none());
        let mut walker = DirectoryWalker::new(
            &empty_source,
            empty.footer,
            empty.summary,
            limits(),
            &mut [],
            &mut [],
        )
        .unwrap();
        assert_eq!(walker.finish_validation().unwrap(), WalkSummary::default());
        assert_eq!(empty_source.calls.get(), 0);
    }

    #[test]
    fn directory_maximum_depth_uses_exactly_eight_caller_slots() {
        let (mut bytes, payload) = start_image();
        let mut page = leaf(&mut bytes, key(0), payload);
        for depth in 2..=MAX_DEPTH {
            page = parent(
                &mut bytes,
                depth,
                1,
                &[InteriorEntry {
                    lower: key(0),
                    upper: key(0),
                    child: page,
                }],
            );
        }
        let image = seal(
            bytes,
            DirectoryRoot {
                depth: MAX_DEPTH,
                page,
            },
            1,
        );
        let source = Source::new(&image.bytes);
        let requirements =
            DirectoryBufferRequirements::new(&image.footer, &image.summary, &limits()).unwrap();
        assert_eq!(requirements.walker_slots, 8);
        assert_eq!(requirements.walker_node_bytes, 8 * MAX_NODE_BYTES);
        let mut nodes = [[0; MAX_NODE_BYTES]; MAX_DEPTH as usize];
        assert!(matches!(
            DirectoryWalker::new(
                &source,
                image.footer,
                image.summary,
                limits(),
                &mut nodes[..7],
                &mut []
            ),
            Err(DirectoryReadError::BufferTooSmall)
        ));
        assert_eq!(source.calls.get(), 0);
        let mut lookup_node = [0; MAX_NODE_BYTES];
        let mut lookup = DirectoryLookup::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut lookup_node,
            &mut [],
        )
        .unwrap();
        assert!(lookup.find(lookup_key(0)).unwrap().is_some());
        assert_eq!(source.calls.get(), 8);
        let mut walker = DirectoryWalker::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut nodes,
            &mut [],
        )
        .unwrap();
        assert_eq!(
            walker.finish_validation().unwrap(),
            WalkSummary {
                entry_count: 1,
                node_count: 8,
                depth: 8
            }
        );
        assert_eq!(source.calls.get(), 16);
    }

    #[test]
    fn directory_original_io_error_and_caught_read_panic_are_retained() {
        let image = build(65);
        let source = Source {
            fail_call: 2,
            ..Source::new(&image.bytes)
        };
        let mut node = [0; MAX_NODE_BYTES];
        let mut lookup = DirectoryLookup::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut node,
            &mut [],
        )
        .unwrap();
        assert!(
            matches!(lookup.find(lookup_key(0)), Err(DirectoryReadError::PageIo(PageIoError::Io(error))) if error.kind() == io::ErrorKind::PermissionDenied)
        );
        assert!(lookup.find(lookup_key(0)).is_err());
        assert_eq!(source.calls.get(), 2);
        let source = Source {
            panic_call: 2,
            ..Source::new(&image.bytes)
        };
        let mut nodes = [[0; MAX_NODE_BYTES]; 2];
        let mut walker = DirectoryWalker::new(
            &source,
            image.footer,
            image.summary,
            limits(),
            &mut nodes,
            &mut [],
        )
        .unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            walker.next_entry().unwrap();
        }));
        assert!(panic.is_err());
        assert!(matches!(
            walker.next_entry(),
            Err(DirectoryReadError::InterruptedOperation)
        ));
        assert!(matches!(
            walker.finish_validation(),
            Err(DirectoryReadError::InterruptedOperation)
        ));
        assert_eq!(source.calls.get(), 2);
    }
}
