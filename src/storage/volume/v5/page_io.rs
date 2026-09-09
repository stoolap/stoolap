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

//! Append-only page I/O without file or buffer ownership. The caller owns the
//! immutable file/FD lease, memory reservations, and buffers. This module never
//! allocates a read buffer or retains a collection of page descriptors.
//!
//! Readers first obtain a validated PageReadPlan, reserve its reported bytes,
//! and only then allocate/borrow buffers and issue I/O. Stored CRC is checked
//! before Raw access or LZ4 decompression; decompression receives exactly the
//! declared output slice. Root decoding is fixed at 128 bytes and directory
//! nodes are capped at 8 KiB before the caller reserves decoded memory.
//!
//! OpenedEnvelope reads only the fixed header/footer. Its identity must be
//! compared against the manifest's expected identity before trusting this file;
//! use require_identity. File length and contents must remain immutable during
//! all reads. ReadAt is deliberately generic so the engine's existing FD owner
//! can provide positioned access without a second file-lifetime implementation.
//!
//! PageWriter borrows an empty sink positioned at offset zero. It appends a
//! header, payload/directory pages, typed root summary, then footer. Checked
//! offsets are logical offsets in that sink. Semantic preflight failures leave
//! the sink unchanged; an I/O failure permanently poisons the writer because a
//! prefix may have reached storage. Success returns metadata only. The caller
//! remains responsible for flush/sync, durable manifest publication and cleanup.

use std::fmt;
use std::io::{self, Write};

use super::directory::{
    encode_interior, encode_leaf, DirectoryError, DirectoryRoot, InteriorEntry, LeafEntry,
    RootSummary, MAX_NODE_BYTES, ROOT_SUMMARY_BYTES,
};
use super::envelope::{
    Codec, EnvelopeError, FileIdentity, Footer, Header, PageDescriptor, ReadLimits, FOOTER_SIZE,
    HEADER_SIZE,
};

/// Positioned access supplied by the caller's existing file owner. Like Read,
/// this may return short reads or Interrupted; it must never return more than
/// dst.len(). It must not change an implicit stream cursor.
pub trait ReadAt {
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize>;
}

#[derive(Debug)]
pub enum PageIoError {
    Envelope(EnvelopeError),
    Directory(DirectoryError),
    Io(io::Error),
    IdentityMismatch,
    AddressSpace,
    BufferTooSmall,
    UnexpectedEof,
    InvalidReadCount,
    InvalidLz4,
    DecodedLengthMismatch,
    RootLength,
    DirectoryTooLarge,
    RootMismatch,
    PayloadLength,
    Poisoned,
    Finished,
}
impl fmt::Display for PageIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Envelope(error) => write!(f, "V5 page I/O: {error}"),
            Self::Directory(error) => write!(f, "V5 page I/O: {error}"),
            Self::Io(error) => write!(f, "V5 page I/O: {error}"),
            other => write!(f, "invalid V5 page I/O: {other:?}"),
        }
    }
}
impl std::error::Error for PageIoError {}
impl From<EnvelopeError> for PageIoError {
    fn from(error: EnvelopeError) -> Self {
        Self::Envelope(error)
    }
}
impl From<DirectoryError> for PageIoError {
    fn from(error: DirectoryError) -> Self {
        Self::Directory(error)
    }
}
type Result<T> = std::result::Result<T, PageIoError>;

#[derive(Clone, Copy, Debug)]
pub struct OpenedEnvelope {
    pub header: Header,
    pub footer: Footer,
}
impl OpenedEnvelope {
    /// Read only the two fixed envelopes. The caller must compare the returned
    /// identity with its manifest record before following any page pointer.
    pub fn read<R: ReadAt + ?Sized>(
        source: &R,
        file_length: u64,
        limits: &ReadLimits,
    ) -> Result<Self> {
        if file_length <= (HEADER_SIZE + FOOTER_SIZE) as u64 {
            return Err(EnvelopeError::FileTooShort.into());
        }
        let mut header_bytes = [0; HEADER_SIZE];
        let mut footer_bytes = [0; FOOTER_SIZE];
        read_exact_at(source, 0, &mut header_bytes)?;
        let header = Header::decode(&header_bytes)?;
        read_exact_at(source, file_length - FOOTER_SIZE as u64, &mut footer_bytes)?;
        let footer = Footer::decode(&footer_bytes, file_length, limits)?;
        // Do not let an invalid typed root size reach a later buffer reservation.
        if footer.root.decoded_len != ROOT_SUMMARY_BYTES as u64 {
            return Err(PageIoError::RootLength);
        }
        Ok(Self { header, footer })
    }
    pub fn require_identity(&self, expected: FileIdentity) -> Result<()> {
        if self.header.identity != expected {
            Err(PageIoError::IdentityMismatch)
        } else {
            Ok(())
        }
    }
}

/// Validated scalar sizes and address. Construct before reserving/allocating
/// buffers; private fields prevent callers from bypassing length checks.
#[derive(Clone, Copy, Debug)]
pub struct PageReadPlan {
    descriptor: PageDescriptor,
    stored_len: usize,
    decoded_len: usize,
}
impl PageReadPlan {
    pub fn for_root(footer: &Footer, limits: &ReadLimits) -> Result<Self> {
        footer.validate_root(limits)?;
        if footer.root.decoded_len != ROOT_SUMMARY_BYTES as u64 {
            return Err(PageIoError::RootLength);
        }
        Self::from_validated(footer.root)
    }
    pub fn for_page(
        footer: &Footer,
        descriptor: PageDescriptor,
        limits: &ReadLimits,
    ) -> Result<Self> {
        footer.validate_page(&descriptor, limits)?;
        Self::from_validated(descriptor)
    }
    pub fn for_directory(
        footer: &Footer,
        descriptor: PageDescriptor,
        limits: &ReadLimits,
    ) -> Result<Self> {
        footer.validate_page(&descriptor, limits)?;
        if descriptor.decoded_len > MAX_NODE_BYTES as u64 {
            return Err(PageIoError::DirectoryTooLarge);
        }
        Self::from_validated(descriptor)
    }
    fn from_validated(descriptor: PageDescriptor) -> Result<Self> {
        let stored_len =
            usize::try_from(descriptor.stored_len).map_err(|_| PageIoError::AddressSpace)?;
        let decoded_len =
            usize::try_from(descriptor.decoded_len).map_err(|_| PageIoError::AddressSpace)?;
        // Check combined reservation arithmetic as well as individual buffers.
        if descriptor.codec != Codec::Raw {
            stored_len
                .checked_add(decoded_len)
                .ok_or(PageIoError::AddressSpace)?;
        }
        Ok(Self {
            descriptor,
            stored_len,
            decoded_len,
        })
    }
    pub const fn descriptor(&self) -> PageDescriptor {
        self.descriptor
    }
    pub const fn stored_buffer_len(&self) -> usize {
        self.stored_len
    }
    /// Raw pages alias their stored buffer and need no second buffer.
    pub const fn decoded_buffer_len(&self) -> usize {
        match self.descriptor.codec {
            Codec::Raw => 0,
            Codec::Lz4Block => self.decoded_len,
        }
    }
    pub fn read_into<'a, R: ReadAt + ?Sized>(
        &self,
        source: &R,
        stored: &'a mut [u8],
        decoded: &'a mut [u8],
    ) -> Result<&'a [u8]> {
        if stored.len() < self.stored_len || decoded.len() < self.decoded_buffer_len() {
            return Err(PageIoError::BufferTooSmall);
        }
        let stored = &mut stored[..self.stored_len];
        read_exact_at(source, self.descriptor.offset, stored)?;
        self.descriptor.verify_stored_bytes(stored)?;
        match self.descriptor.codec {
            Codec::Raw => Ok(stored),
            Codec::Lz4Block => {
                // A larger caller buffer cannot legitimize output beyond the
                // checked descriptor or permit hidden bytes after decoded_len.
                let decoded = &mut decoded[..self.decoded_len];
                let produced = lz4_flex::block::decompress_into(stored, decoded)
                    .map_err(|_| PageIoError::InvalidLz4)?;
                if produced != self.decoded_len {
                    return Err(PageIoError::DecodedLengthMismatch);
                }
                Ok(decoded)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriterState {
    Open,
    Poisoned,
    Finished,
}

pub struct PageWriter<'a, W: Write + ?Sized> {
    sink: &'a mut W,
    header: Header,
    limits: ReadLimits,
    next_offset: u64,
    state: WriterState,
    last_directory: Option<(DirectoryRoot, u64)>,
}
#[derive(Clone, Copy, Debug)]
pub struct FinishedVolume {
    pub identity: FileIdentity,
    pub footer: Footer,
    pub summary: RootSummary,
}
impl<'a, W: Write + ?Sized> PageWriter<'a, W> {
    /// Precondition: sink is empty and positioned at offset zero. No Seek or
    /// truncate is performed here. If the initial header write fails, discard
    /// this unpublishable sink; no writer is returned for further appends.
    pub fn new(sink: &'a mut W, header: Header, limits: ReadLimits) -> Result<Self> {
        let bytes = header.encode()?;
        if limits.root_stored_bytes < ROOT_SUMMARY_BYTES as u64 {
            return Err(EnvelopeError::StoredLimitExceeded.into());
        }
        if limits.root_decoded_bytes < ROOT_SUMMARY_BYTES as u64 {
            return Err(EnvelopeError::DecodedLimitExceeded.into());
        }
        let mut writer = Self {
            sink,
            header,
            limits,
            next_offset: 0,
            state: WriterState::Open,
            last_directory: None,
        };
        writer.write_bytes(&bytes)?;
        Ok(writer)
    }
    pub const fn position(&self) -> u64 {
        self.next_offset
    }
    pub const fn limits(&self) -> ReadLimits {
        self.limits
    }
    pub const fn is_poisoned(&self) -> bool {
        matches!(self.state, WriterState::Poisoned)
    }
    /// Append an already encoded page. The caller owns compression scratch and
    /// must supply the true decoded length for LZ4; decode validation on read is
    /// still mandatory. Descriptor bytes describe only this successful append.
    pub fn append_stored(
        &mut self,
        codec: Codec,
        stored: &[u8],
        decoded_len: u64,
    ) -> Result<PageDescriptor> {
        self.check_open()?;
        let descriptor = self.preflight_page(codec, stored, decoded_len)?;
        self.write_bytes(stored)?;
        Ok(descriptor)
    }
    pub fn append_leaf(
        &mut self,
        entries: &[LeafEntry],
        scratch: &mut [u8],
    ) -> Result<PageDescriptor> {
        self.check_open()?;
        let len = encode_leaf(entries, scratch)?;
        for entry in entries {
            self.validate_reference(entry.page)?;
        }
        let page = self.append_stored(Codec::Raw, &scratch[..len], len as u64)?;
        self.last_directory = Some((DirectoryRoot { depth: 1, page }, entries.len() as u64));
        Ok(page)
    }
    pub fn append_interior(
        &mut self,
        depth: u8,
        total: u64,
        entries: &[InteriorEntry],
        scratch: &mut [u8],
    ) -> Result<PageDescriptor> {
        self.check_open()?;
        let len = encode_interior(depth, total, entries, scratch)?;
        for entry in entries {
            self.validate_reference(entry.child)?;
        }
        let page = self.append_stored(Codec::Raw, &scratch[..len], len as u64)?;
        self.last_directory = Some((DirectoryRoot { depth, page }, total));
        Ok(page)
    }
    /// Finish without flush/sync or publication. Validation errors keep the
    /// writer open; I/O errors poison it; success rejects all future writes.
    pub fn finish(&mut self, summary: &RootSummary) -> Result<FinishedVolume> {
        self.check_open()?;
        summary.validate_header(&self.header)?;
        let root_bytes = summary.encode()?;
        match (summary.directory, self.last_directory) {
            (None, None) => (),
            (Some(root), Some((written, count)))
                if root == written && summary.entry_count == count =>
            {
                self.validate_reference(root.page)?;
            }
            _ => return Err(PageIoError::RootMismatch),
        }
        let root = PageDescriptor {
            offset: self.next_offset,
            stored_len: ROOT_SUMMARY_BYTES as u64,
            decoded_len: ROOT_SUMMARY_BYTES as u64,
            stored_checksum: crc32fast::hash(&root_bytes),
            codec: Codec::Raw,
        };
        let file_length = self
            .next_offset
            .checked_add((ROOT_SUMMARY_BYTES + FOOTER_SIZE) as u64)
            .ok_or(EnvelopeError::OffsetOverflow)?;
        let footer = Footer { file_length, root };
        footer.validate_root(&self.limits)?;
        let footer_bytes = footer.encode()?;
        self.write_bytes(&root_bytes)?;
        self.write_bytes(&footer_bytes)?;
        self.state = WriterState::Finished;
        Ok(FinishedVolume {
            identity: self.header.identity,
            footer,
            summary: *summary,
        })
    }
    pub(crate) fn check_open(&self) -> Result<()> {
        match self.state {
            WriterState::Open => Ok(()),
            WriterState::Poisoned => Err(PageIoError::Poisoned),
            WriterState::Finished => Err(PageIoError::Finished),
        }
    }
    fn preflight_page(
        &self,
        codec: Codec,
        stored: &[u8],
        decoded_len: u64,
    ) -> Result<PageDescriptor> {
        let stored_len = u64::try_from(stored.len()).map_err(|_| PageIoError::AddressSpace)?;
        let mut descriptor = PageDescriptor {
            offset: self.next_offset,
            stored_len,
            decoded_len,
            stored_checksum: 0,
            codec,
        };
        descriptor.encode()?;
        if stored_len > self.limits.page_stored_bytes {
            return Err(EnvelopeError::StoredLimitExceeded.into());
        }
        if decoded_len > self.limits.page_decoded_bytes {
            return Err(EnvelopeError::DecodedLimitExceeded.into());
        }
        self.next_offset
            .checked_add(stored_len)
            .and_then(|end| end.checked_add((ROOT_SUMMARY_BYTES + FOOTER_SIZE) as u64))
            .ok_or(EnvelopeError::OffsetOverflow)?;
        descriptor.stored_checksum = crc32fast::hash(stored);
        Ok(descriptor)
    }
    pub(crate) fn validate_reference(&self, page: PageDescriptor) -> Result<()> {
        page.encode()?;
        if page.stored_len > self.limits.page_stored_bytes {
            return Err(EnvelopeError::StoredLimitExceeded.into());
        }
        if page.decoded_len > self.limits.page_decoded_bytes {
            return Err(EnvelopeError::DecodedLimitExceeded.into());
        }
        let end = page
            .offset
            .checked_add(page.stored_len)
            .ok_or(EnvelopeError::OffsetOverflow)?;
        if page.offset < HEADER_SIZE as u64 || end > self.next_offset {
            return Err(DirectoryError::ForwardReference.into());
        }
        Ok(())
    }
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let length = u64::try_from(bytes.len()).map_err(|_| PageIoError::AddressSpace)?;
        let end = self
            .next_offset
            .checked_add(length)
            .ok_or(EnvelopeError::OffsetOverflow)?;
        // Arbitrary Write implementations may unwind after writing a prefix.
        // Set the fail-closed state before entering them so a caller that catches
        // the panic cannot append at the stale pre-write logical position.
        self.state = WriterState::Poisoned;
        if let Err(error) = self.sink.write_all(bytes) {
            return Err(PageIoError::Io(error));
        }
        self.next_offset = end;
        self.state = WriterState::Open;
        Ok(())
    }
}

fn read_exact_at<R: ReadAt + ?Sized>(
    source: &R,
    mut offset: u64,
    mut dst: &mut [u8],
) -> Result<()> {
    let length = u64::try_from(dst.len()).map_err(|_| PageIoError::AddressSpace)?;
    offset
        .checked_add(length)
        .ok_or(EnvelopeError::OffsetOverflow)?;
    while !dst.is_empty() {
        match source.read_at(offset, dst) {
            Ok(0) => return Err(PageIoError::UnexpectedEof),
            Ok(n) if n > dst.len() => return Err(PageIoError::InvalidReadCount),
            Ok(n) => {
                offset += n as u64;
                dst = &mut dst[n..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => (),
            Err(error) => return Err(PageIoError::Io(error)),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::directory::{
        DirectoryKey, DirectoryNode, Layout, RowBounds, Section, KEY_REQUIRED,
    };
    use super::*;
    use std::cell::Cell;

    fn limits() -> ReadLimits {
        ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: 16384,
            page_decoded_bytes: 16384,
        }
    }
    fn identity() -> FileIdentity {
        FileIdentity::new(1, 1, 2).unwrap()
    }
    fn empty() -> RootSummary {
        RootSummary {
            layout: Layout::RowId,
            legacy_base: None,
            row_count: 0,
            column_count: 0,
            group_count: 0,
            entry_count: 0,
            rows: None,
            window: None,
            directory: None,
        }
    }
    fn key(ordinal: u64) -> DirectoryKey {
        DirectoryKey {
            section: Section::SourceLsns as u16,
            flags: KEY_REQUIRED,
            column: u32::MAX,
            ordinal,
        }
    }
    struct SliceReader<'a> {
        bytes: &'a [u8],
        max_read: usize,
        calls: Cell<usize>,
        interrupt: Cell<bool>,
        bytes_read: Cell<usize>,
    }
    impl<'a> SliceReader<'a> {
        fn new(bytes: &'a [u8]) -> Self {
            Self {
                bytes,
                max_read: usize::MAX,
                calls: Cell::new(0),
                interrupt: Cell::new(false),
                bytes_read: Cell::new(0),
            }
        }
    }
    impl ReadAt for SliceReader<'_> {
        fn read_at(&self, offset: u64, dst: &mut [u8]) -> io::Result<usize> {
            self.calls.set(self.calls.get() + 1);
            if self.interrupt.replace(false) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let Some(tail) = usize::try_from(offset)
                .ok()
                .and_then(|p| self.bytes.get(p..))
            else {
                return Ok(0);
            };
            let n = tail.len().min(dst.len()).min(self.max_read);
            dst[..n].copy_from_slice(&tail[..n]);
            self.bytes_read.set(self.bytes_read.get() + n);
            Ok(n)
        }
    }
    fn raw_page(offset: u64, bytes: &[u8]) -> PageDescriptor {
        PageDescriptor {
            offset,
            stored_len: bytes.len() as u64,
            decoded_len: bytes.len() as u64,
            stored_checksum: crc32fast::hash(bytes),
            codec: Codec::Raw,
        }
    }
    fn test_footer() -> Footer {
        Footer {
            file_length: 16384 + 128 + 64,
            root: raw_page(16384, &[0; 128]),
        }
    }

    #[test]
    fn page_io_empty_file_and_bounded_envelope_startup() {
        let mut bytes = Vec::new();
        let mut writer = PageWriter::new(&mut bytes, Header::new(identity()), limits()).unwrap();
        let done = writer.finish(&empty()).unwrap();
        assert_eq!(writer.position(), 256);
        assert!(matches!(
            writer.append_stored(Codec::Raw, b"x", 1),
            Err(PageIoError::Finished)
        ));
        assert!(matches!(
            writer.finish(&empty()),
            Err(PageIoError::Finished)
        ));
        assert_eq!(done.footer.file_length, 256);
        assert_eq!(bytes.len(), 256);
        // Independently packed Python struct/zlib image of the complete empty
        // file, including both envelope checksums and the typed summary CRC.
        assert_eq!(crc32fast::hash(&bytes), 0xd5ed_66c0);
        let source = SliceReader::new(&bytes);
        let opened = OpenedEnvelope::read(&source, bytes.len() as u64, &limits()).unwrap();
        assert_eq!(source.calls.get(), 2);
        assert_eq!(source.bytes_read.get(), 128); // The root is not eagerly fetched.
        opened.require_identity(identity()).unwrap();
        assert!(matches!(
            opened.require_identity(FileIdentity::new(1, 2, 2).unwrap()),
            Err(PageIoError::IdentityMismatch)
        ));
        let plan = PageReadPlan::for_root(&opened.footer, &limits()).unwrap();
        assert_eq!(
            (plan.stored_buffer_len(), plan.decoded_buffer_len()),
            (128, 0)
        );
        let mut storage = [0; 128];
        let root_bytes = plan.read_into(&source, &mut storage, &mut []).unwrap();
        assert_eq!(
            RootSummary::decode(root_bytes, &opened.footer, &limits()).unwrap(),
            empty()
        );
        assert_eq!(source.bytes_read.get(), 256);
    }

    #[test]
    fn page_io_two_level_file_round_trip_and_exact_offsets() {
        let mut bytes = Vec::new();
        let mut scratch = [0; MAX_NODE_BYTES];
        let mut writer = PageWriter::new(&mut bytes, Header::new(identity()), limits()).unwrap();
        let payloads = [
            writer.append_stored(Codec::Raw, b"first000", 8).unwrap(),
            writer.append_stored(Codec::Raw, b"second00", 8).unwrap(),
        ];
        let left = writer
            .append_leaf(
                &[LeafEntry {
                    key: key(0),
                    page: payloads[0],
                }],
                &mut scratch,
            )
            .unwrap();
        let right = writer
            .append_leaf(
                &[LeafEntry {
                    key: key(1),
                    page: payloads[1],
                }],
                &mut scratch,
            )
            .unwrap();
        let root_page = writer
            .append_interior(
                2,
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
                &mut scratch,
            )
            .unwrap();
        assert_eq!(
            (
                payloads[0].offset,
                payloads[1].offset,
                left.offset,
                right.offset,
                root_page.offset
            ),
            (64, 72, 80, 160, 240)
        );
        let summary = RootSummary {
            row_count: 2,
            group_count: 2,
            entry_count: 2,
            rows: Some(RowBounds {
                min: i64::MIN,
                max: i64::MAX,
            }),
            directory: Some(DirectoryRoot {
                depth: 2,
                page: root_page,
            }),
            ..empty()
        };
        let done = writer.finish(&summary).unwrap();
        assert_eq!(done.footer.file_length, 592);
        let source = SliceReader {
            max_read: 3,
            interrupt: Cell::new(true),
            ..SliceReader::new(&bytes)
        };
        let opened = OpenedEnvelope::read(&source, bytes.len() as u64, &limits()).unwrap();
        let mut root_storage = [0; 128];
        let root_bytes = PageReadPlan::for_root(&opened.footer, &limits())
            .unwrap()
            .read_into(&source, &mut root_storage, &mut [])
            .unwrap();
        let summary = RootSummary::decode(root_bytes, &opened.footer, &limits()).unwrap();
        let mut parent_storage = [0; MAX_NODE_BYTES];
        let parent_bytes = PageReadPlan::for_directory(&opened.footer, root_page, &limits())
            .unwrap()
            .read_into(&source, &mut parent_storage, &mut [])
            .unwrap();
        let parent =
            DirectoryNode::decode(parent_bytes, root_page, &opened.footer, &limits()).unwrap();
        summary.validate_directory_root(&parent).unwrap();
        let mut child_storage = [0; MAX_NODE_BYTES];
        let mut payload_storage = [0; 8];
        for (ordinal, descriptor) in [left, right].into_iter().enumerate() {
            let child_bytes = PageReadPlan::for_directory(&opened.footer, descriptor, &limits())
                .unwrap()
                .read_into(&source, &mut child_storage, &mut [])
                .unwrap();
            let child =
                DirectoryNode::decode(child_bytes, descriptor, &opened.footer, &limits()).unwrap();
            parent.validate_child(ordinal, &child).unwrap();
            let payload = child.leaves().next().unwrap().unwrap().page;
            let value = PageReadPlan::for_page(&opened.footer, payload, &limits())
                .unwrap()
                .read_into(&source, &mut payload_storage, &mut [])
                .unwrap();
            assert_eq!(
                value,
                if ordinal == 0 {
                    b"first000"
                } else {
                    b"second00"
                }
            );
        }
        assert!(source.calls.get() > 20); // Short reads and Interrupted were actually exercised.
    }

    #[test]
    fn page_io_preflight_limits_reject_before_read_or_buffer_access() {
        let footer = test_footer();
        let source = SliceReader::new(&[]);
        let page = raw_page(64, b"12345678");
        let mut stored = [0xa5; 8];
        let mut decoded = [0xb6; 8];
        let plan = PageReadPlan::for_page(&footer, page, &limits()).unwrap();
        assert!(matches!(
            plan.read_into(&source, &mut stored[..7], &mut decoded),
            Err(PageIoError::BufferTooSmall)
        ));
        assert_eq!(source.calls.get(), 0);
        assert_eq!(stored, [0xa5; 8]);
        assert_eq!(decoded, [0xb6; 8]);
        for bad in [
            PageDescriptor {
                offset: u64::MAX,
                ..page
            },
            PageDescriptor {
                stored_len: u64::MAX,
                decoded_len: u64::MAX,
                ..page
            },
            PageDescriptor { offset: 63, ..page },
        ] {
            assert!(PageReadPlan::for_page(&footer, bad, &limits()).is_err());
        }
        let oversized = PageDescriptor {
            stored_len: 1,
            decoded_len: MAX_NODE_BYTES as u64 + 1,
            codec: Codec::Lz4Block,
            ..page
        };
        assert!(matches!(
            PageReadPlan::for_directory(&footer, oversized, &limits()),
            Err(PageIoError::DirectoryTooLarge)
        ));
        let bad_root = Footer {
            root: PageDescriptor {
                stored_len: 128,
                decoded_len: 129,
                codec: Codec::Lz4Block,
                ..footer.root
            },
            ..footer
        };
        let larger = ReadLimits {
            root_decoded_bytes: 1000,
            ..limits()
        };
        assert!(matches!(
            PageReadPlan::for_root(&bad_root, &larger),
            Err(PageIoError::RootLength)
        ));
        assert_eq!(source.calls.get(), 0);
    }

    #[test]
    fn page_io_crc_precedes_decompression_and_exact_output_is_enforced() {
        let value = b"same same same same same same same same";
        let compressed = lz4_flex::block::compress(value);
        let mut file = vec![0; 64];
        file.extend_from_slice(&compressed);
        let descriptor = PageDescriptor {
            offset: 64,
            stored_len: compressed.len() as u64,
            decoded_len: value.len() as u64,
            stored_checksum: crc32fast::hash(&compressed),
            codec: Codec::Lz4Block,
        };
        let source = SliceReader::new(&file);
        let mut stored = [0xa5; 128];
        let mut decoded = [0xb6; 128];
        let plan = PageReadPlan::for_page(&test_footer(), descriptor, &limits()).unwrap();
        assert_eq!(
            plan.read_into(&source, &mut stored, &mut decoded).unwrap(),
            value
        );
        assert!(stored[compressed.len()..].iter().all(|&v| v == 0xa5));
        assert!(decoded[value.len()..].iter().all(|&v| v == 0xb6));
        // Caller offers ample space, but the descriptor's exact destination is
        // smaller/larger; neither oversized output nor a short result is valid.
        let shorter = PageReadPlan::for_page(
            &test_footer(),
            PageDescriptor {
                decoded_len: descriptor.decoded_len - 1,
                ..descriptor
            },
            &limits(),
        )
        .unwrap();
        assert!(matches!(
            shorter.read_into(&source, &mut stored, &mut decoded),
            Err(PageIoError::InvalidLz4)
        ));
        let longer = PageReadPlan::for_page(
            &test_footer(),
            PageDescriptor {
                decoded_len: descriptor.decoded_len + 1,
                ..descriptor
            },
            &limits(),
        )
        .unwrap();
        assert!(matches!(
            longer.read_into(&source, &mut stored, &mut decoded),
            Err(PageIoError::DecodedLengthMismatch)
        ));
        let bad_crc = PageReadPlan::for_page(
            &test_footer(),
            PageDescriptor {
                stored_checksum: 0,
                ..descriptor
            },
            &limits(),
        )
        .unwrap();
        decoded.fill(0xb6);
        assert!(matches!(
            bad_crc.read_into(&source, &mut stored, &mut decoded),
            Err(PageIoError::Envelope(EnvelopeError::ChecksumMismatch))
        ));
        assert!(decoded.iter().all(|&v| v == 0xb6));
        // A malformed stream with a correct stored checksum must still fail.
        let malformed = [0xff];
        file.truncate(64);
        file.extend_from_slice(&malformed);
        let source = SliceReader::new(&file);
        let bad_lz4 = PageReadPlan::for_page(
            &test_footer(),
            PageDescriptor {
                stored_len: 1,
                stored_checksum: crc32fast::hash(&malformed),
                ..descriptor
            },
            &limits(),
        )
        .unwrap();
        assert!(matches!(
            bad_lz4.read_into(&source, &mut stored, &mut decoded),
            Err(PageIoError::InvalidLz4)
        ));
    }

    #[test]
    fn page_io_truncation_never_returns_successful_partial_bytes() {
        let mut bytes = vec![0; 64];
        bytes.extend_from_slice(b"1234567");
        let source = SliceReader {
            max_read: 2,
            ..SliceReader::new(&bytes)
        };
        let plan =
            PageReadPlan::for_page(&test_footer(), raw_page(64, b"12345678"), &limits()).unwrap();
        let mut stored = [0; 8];
        assert!(matches!(
            plan.read_into(&source, &mut stored, &mut []),
            Err(PageIoError::UnexpectedEof)
        ));
        assert_eq!(source.bytes_read.get(), 7);
        assert!(matches!(
            OpenedEnvelope::read(&source, 128, &limits()),
            Err(PageIoError::Envelope(EnvelopeError::FileTooShort))
        ));
        struct ImpossibleReader;
        impl ReadAt for ImpossibleReader {
            fn read_at(&self, _: u64, dst: &mut [u8]) -> io::Result<usize> {
                Ok(dst.len() + 1)
            }
        }
        assert!(matches!(
            plan.read_into(&ImpossibleReader, &mut stored, &mut []),
            Err(PageIoError::InvalidReadCount)
        ));
    }

    struct FailingWriter {
        bytes: Vec<u8>,
        limit: usize,
        calls: usize,
    }
    impl Write for FailingWriter {
        fn write(&mut self, src: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            let n = self.limit.saturating_sub(self.bytes.len()).min(src.len());
            if n == 0 {
                return Err(io::ErrorKind::Other.into());
            }
            self.bytes.extend_from_slice(&src[..n]);
            Ok(n)
        }
        fn flush(&mut self) -> io::Result<()> {
            panic!("page writer must not flush/sync caller sink")
        }
    }

    #[test]
    fn page_io_partial_writes_poison_writer_including_root_and_footer() {
        for limit in [66, 70, 64 + 128 + 5] {
            let mut sink = FailingWriter {
                bytes: Vec::new(),
                limit,
                calls: 0,
            };
            let mut writer = PageWriter::new(&mut sink, Header::new(identity()), limits()).unwrap();
            let result = if limit == 66 {
                writer.append_stored(Codec::Raw, b"12345678", 8).map(|_| ())
            } else {
                writer.finish(&empty()).map(|_| ())
            };
            assert!(matches!(result, Err(PageIoError::Io(_))));
            assert!(writer.is_poisoned());
            assert!(matches!(
                writer.append_stored(Codec::Raw, b"x", 1),
                Err(PageIoError::Poisoned)
            ));
            assert!(matches!(
                writer.finish(&empty()),
                Err(PageIoError::Poisoned)
            ));
            let position = writer.position();
            assert_eq!(sink.bytes.len(), limit);
            assert_eq!(sink.calls, if limit > 192 { 4 } else { 3 });
            assert_eq!(position, if limit > 192 { 192 } else { 64 });
        }
        let mut sink = FailingWriter {
            bytes: Vec::new(),
            limit: 3,
            calls: 0,
        };
        assert!(matches!(
            PageWriter::new(&mut sink, Header::new(identity()), limits()),
            Err(PageIoError::Io(_))
        ));
        assert_eq!(sink.bytes.len(), 3);
    }

    #[test]
    fn page_io_caught_sink_panic_keeps_writer_poisoned() {
        struct PanicWriter {
            bytes: Vec<u8>,
            calls: usize,
        }
        impl Write for PanicWriter {
            fn write(&mut self, source: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                if self.calls == 2 {
                    self.bytes.extend_from_slice(&source[..2]);
                    panic!("injected panic after partial payload write");
                }
                self.bytes.extend_from_slice(source);
                Ok(source.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut sink = PanicWriter {
            bytes: Vec::new(),
            calls: 0,
        };
        let mut writer = PageWriter::new(&mut sink, Header::new(identity()), limits()).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            writer.append_stored(Codec::Raw, b"12345678", 8).unwrap();
        }));
        assert!(panic.is_err());
        assert!(writer.is_poisoned());
        assert!(matches!(
            writer.append_stored(Codec::Raw, b"x", 1),
            Err(PageIoError::Poisoned)
        ));
        assert!(matches!(
            writer.finish(&empty()),
            Err(PageIoError::Poisoned)
        ));
        assert_eq!(sink.bytes.len(), 66);
        assert_eq!(sink.calls, 2);
    }

    #[test]
    fn page_io_checked_addresses_include_both_buffers_and_final_envelope() {
        let huge_limits = ReadLimits {
            root_stored_bytes: u64::MAX,
            root_decoded_bytes: u64::MAX,
            page_stored_bytes: u64::MAX,
            page_decoded_bytes: u64::MAX,
        };
        let footer = Footer {
            file_length: u64::MAX,
            root: raw_page(u64::MAX - 192, &[0; 128]),
        };
        let huge = PageDescriptor {
            offset: 64,
            stored_len: u64::MAX / 2,
            decoded_len: u64::MAX,
            stored_checksum: 0,
            codec: Codec::Lz4Block,
        };
        assert!(matches!(
            PageReadPlan::for_page(&footer, huge, &huge_limits),
            Err(PageIoError::AddressSpace)
        ));
        let mut bytes = Vec::new();
        let mut writer = PageWriter::new(&mut bytes, Header::new(identity()), limits()).unwrap();
        // Model an enormous append-only sink without allocating its contents.
        writer.next_offset = u64::MAX - 200;
        assert!(matches!(
            writer.append_stored(Codec::Raw, &[0; 16], 16),
            Err(PageIoError::Envelope(EnvelopeError::OffsetOverflow))
        ));
        assert!(!writer.is_poisoned());
        assert_eq!(writer.position(), u64::MAX - 200);
        assert_eq!(bytes.len(), HEADER_SIZE);
    }

    #[test]
    fn page_io_writer_preflight_errors_preserve_position_and_allow_retry() {
        let mut bytes = Vec::new();
        let mut writer = PageWriter::new(&mut bytes, Header::new(identity()), limits()).unwrap();
        assert!(matches!(
            writer.append_stored(Codec::Raw, b"x", 2),
            Err(PageIoError::Envelope(EnvelopeError::RawLengthMismatch))
        ));
        assert!(matches!(
            writer.append_stored(Codec::Lz4Block, b"x", 16385),
            Err(PageIoError::Envelope(EnvelopeError::DecodedLimitExceeded))
        ));
        let mut scratch = [0; MAX_NODE_BYTES];
        let forward = LeafEntry {
            key: key(0),
            page: raw_page(64, b"12345678"),
        };
        assert!(matches!(
            writer.append_leaf(&[forward], &mut scratch),
            Err(PageIoError::Directory(DirectoryError::ForwardReference))
        ));
        assert_eq!(writer.position(), 64);
        assert!(!writer.is_poisoned());
        let data = writer.append_stored(Codec::Raw, b"12345678", 8).unwrap();
        let leaf = writer
            .append_leaf(
                &[LeafEntry {
                    key: key(0),
                    page: data,
                }],
                &mut scratch,
            )
            .unwrap();
        assert!(matches!(
            writer.finish(&empty()),
            Err(PageIoError::RootMismatch)
        ));
        let summary = RootSummary {
            row_count: 1,
            group_count: 1,
            entry_count: 1,
            rows: Some(RowBounds { min: 0, max: 0 }),
            directory: Some(DirectoryRoot {
                depth: 1,
                page: leaf,
            }),
            ..empty()
        };
        let mismatch = RootSummary {
            directory: Some(DirectoryRoot {
                depth: 1,
                page: data,
            }),
            ..summary
        };
        assert!(matches!(
            writer.finish(&mismatch),
            Err(PageIoError::RootMismatch)
        ));
        assert_eq!(writer.position(), 152);
        writer.finish(&summary).unwrap();
        assert_eq!(bytes.len(), 344);
    }
}
