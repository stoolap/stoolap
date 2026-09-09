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

//! Bounded external sorting for directory descriptors emitted in payload order.
//!
//! The caller owns two distinct scratch streams, their lifetime/accounting, and
//! a fixed initial sorting slice. Runs have deterministic lengths, so their
//! count needs no descriptor vector. Each merge pass reads two bounded buffers
//! and rewrites the other stream from offset zero. Old tail bytes are ignored:
//! each stream requires at most 52 * entry_count bytes, independent of passes.
//! This module never opens, truncates, renames, or removes a file.
//!
//! These are ephemeral, single-build records, not a persistent format. Every
//! 52-byte record is a V5 leaf entry followed by its CRC32. The returned run
//! metadata belongs to the exact scratch contents just written; callers must
//! not substitute another stream or alias source and destination backing files.
//! Finished sorted records feed DirectoryWriter after all payloads are written.
//! No successful operation allocates; all buffers remain caller owned.

use std::fmt;
use std::io::{self, Seek, SeekFrom, Write};

use super::directory::{DirectoryError, DirectoryKey, LeafEntry, LEAF_ENTRY_BYTES};
use super::directory_writer::MAX_ENTRIES;
use super::envelope::EnvelopeError;
use super::page_io::ReadAt;

pub const RECORD_BYTES: usize = LEAF_ENTRY_BYTES + 4;
pub const IO_RECORDS: usize = 64;
pub const IO_BYTES: usize = IO_RECORDS * RECORD_BYTES;

#[derive(Debug)]
pub enum RunError {
    Directory(DirectoryError),
    Envelope(EnvelopeError),
    Io(io::Error),
    ScratchTooSmall,
    Capacity,
    Checksum,
    KeyOrder,
    UnexpectedEof,
    InvalidReadCount,
    NotSorted,
    AlreadySorted,
    Poisoned,
    Finished,
}
impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V5 metadata run: {self:?}")
    }
}
impl std::error::Error for RunError {}
impl From<io::Error> for RunError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
impl From<DirectoryError> for RunError {
    fn from(error: DirectoryError) -> Self {
        Self::Directory(error)
    }
}
impl From<EnvelopeError> for RunError {
    fn from(error: EnvelopeError) -> Self {
        Self::Envelope(error)
    }
}
type Result<T> = std::result::Result<T, RunError>;

/// Exact logical contents, excluding any stale bytes beyond byte_len().
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorRuns {
    entries: u64,
    run_len: u64,
}
impl DescriptorRuns {
    pub fn entry_count(self) -> u64 {
        self.entries
    }
    pub fn byte_len(self) -> u64 {
        self.entries * RECORD_BYTES as u64
    }
    pub fn is_sorted(self) -> bool {
        self.entries <= self.run_len
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Open,
    Poisoned,
    Finished,
}

pub struct RunWriter<'a, W: Write + ?Sized> {
    sink: &'a mut W,
    entries: &'a mut [LeafEntry],
    output: &'a mut [u8],
    len: usize,
    count: u64,
    state: State,
}
impl<'a, W: Write + Seek + ?Sized> RunWriter<'a, W> {
    /// Rewinds the caller's exclusive scratch stream, without truncating it.
    /// Semantic preflight precedes seeking and any scratch/output mutation.
    pub fn new(
        sink: &'a mut W,
        entries: &'a mut [LeafEntry],
        output: &'a mut [u8],
    ) -> Result<Self> {
        if entries.is_empty() || output.len() < RECORD_BYTES {
            return Err(RunError::ScratchTooSmall);
        }
        if entries.len() as u128 > u128::from(MAX_ENTRIES) {
            return Err(RunError::Capacity);
        }
        sink.seek(SeekFrom::Start(0))?;
        Ok(Self {
            sink,
            entries,
            output,
            len: 0,
            count: 0,
            state: State::Open,
        })
    }
}
impl<W: Write + ?Sized> RunWriter<'_, W> {
    pub fn push(&mut self, entry: LeafEntry) -> Result<()> {
        self.check_open()?;
        if self.count == MAX_ENTRIES {
            return Err(RunError::Capacity);
        }
        entry.key.validate()?;
        entry.page.encode()?;
        self.state = State::Poisoned;
        self.entries[self.len] = entry;
        self.len += 1;
        self.count += 1;
        if self.len == self.entries.len() {
            self.flush_run()?;
        }
        self.state = State::Open;
        Ok(())
    }
    pub fn finish(&mut self) -> Result<DescriptorRuns> {
        self.check_open()?;
        self.state = State::Poisoned;
        self.flush_run()?;
        self.sink.flush()?;
        self.state = State::Finished;
        Ok(DescriptorRuns {
            entries: self.count,
            run_len: self.entries.len() as u64,
        })
    }
    fn check_open(&self) -> Result<()> {
        match self.state {
            State::Open => Ok(()),
            State::Poisoned => Err(RunError::Poisoned),
            State::Finished => Err(RunError::Finished),
        }
    }
    fn flush_run(&mut self) -> Result<()> {
        let entries = &mut self.entries[..self.len];
        entries.sort_unstable_by(|a, b| a.key.cmp_identity(&b.key));
        if entries
            .windows(2)
            .any(|pair| pair[0].key.identity() == pair[1].key.identity())
        {
            return Err(RunError::KeyOrder);
        }
        let mut output = RecordOutput::new(self.sink, self.output);
        for &entry in entries.iter() {
            output.push(entry)?;
        }
        output.finish()?;
        self.len = 0;
        Ok(())
    }
}

/// Fixed 9,984-byte capacity; reserve and charge before constructing it.
pub struct MergeScratch {
    left: [u8; IO_BYTES],
    right: [u8; IO_BYTES],
    output: [u8; IO_BYTES],
}
impl MergeScratch {
    pub const fn new() -> Self {
        Self {
            left: [0; IO_BYTES],
            right: [0; IO_BYTES],
            output: [0; IO_BYTES],
        }
    }
}
impl Default for MergeScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge adjacent runs into the other scratch stream. Source and destination
/// must have distinct backing storage. Any error/unwind leaves destination
/// unusable; only the successful return certifies its new logical contents.
pub fn merge_pass<R: ReadAt + ?Sized, W: Write + Seek + ?Sized>(
    source: &R,
    sink: &mut W,
    runs: DescriptorRuns,
    scratch: &mut MergeScratch,
) -> Result<DescriptorRuns> {
    if runs.is_sorted() {
        return Err(RunError::AlreadySorted);
    }
    let next_len = runs
        .run_len
        .checked_mul(2)
        .ok_or(RunError::Capacity)?
        .min(runs.entries);
    sink.seek(SeekFrom::Start(0))?;
    let mut output = RecordOutput::new(sink, &mut scratch.output);
    let mut start = 0;
    while start < runs.entries {
        let middle = start.saturating_add(runs.run_len).min(runs.entries);
        let end = middle.saturating_add(runs.run_len).min(runs.entries);
        let mut left = RunCursor::new(start, middle);
        let mut right = RunCursor::new(middle, end);
        let mut a = left.next(source, &mut scratch.left)?;
        let mut b = right.next(source, &mut scratch.right)?;
        let mut previous: Option<DirectoryKey> = None;
        while a.is_some() || b.is_some() {
            let from_left = match (a, b) {
                (Some(a), Some(b)) => a.key.identity() <= b.key.identity(),
                (Some(_), None) => true,
                _ => false,
            };
            let entry = if from_left {
                a.take().unwrap()
            } else {
                b.take().unwrap()
            };
            if previous.is_some_and(|key| key.identity() >= entry.key.identity()) {
                return Err(RunError::KeyOrder);
            }
            output.push(entry)?;
            previous = Some(entry.key);
            if from_left {
                a = left.next(source, &mut scratch.left)?;
            } else {
                b = right.next(source, &mut scratch.right)?;
            }
        }
        start = end;
    }
    output.finish()?;
    sink.flush()?;
    Ok(DescriptorRuns {
        entries: runs.entries,
        run_len: next_len,
    })
}

/// A complete sorted stream, with one caller-owned read buffer. Early stopping
/// does not certify completion. Exhaustion validates exact count and key order.
pub struct SortedReader<'a, R: ReadAt + ?Sized> {
    source: &'a R,
    buffer: &'a mut [u8],
    cursor: RunCursor,
    state: State,
}
impl<'a, R: ReadAt + ?Sized> SortedReader<'a, R> {
    pub fn new(source: &'a R, runs: DescriptorRuns, buffer: &'a mut [u8]) -> Result<Self> {
        if !runs.is_sorted() {
            return Err(RunError::NotSorted);
        }
        if buffer.len() < RECORD_BYTES {
            return Err(RunError::ScratchTooSmall);
        }
        Ok(Self {
            source,
            buffer,
            cursor: RunCursor::new(0, runs.entries),
            state: State::Open,
        })
    }
    pub fn next_entry(&mut self) -> Result<Option<LeafEntry>> {
        match self.state {
            State::Finished => return Ok(None),
            State::Poisoned => return Err(RunError::Poisoned),
            State::Open => (),
        }
        self.state = State::Poisoned;
        let entry = self.cursor.next(self.source, self.buffer)?;
        self.state = if entry.is_some() {
            State::Open
        } else {
            State::Finished
        };
        Ok(entry)
    }
    pub fn is_complete(&self) -> bool {
        self.state == State::Finished
    }
}

struct RunCursor {
    next: u64,
    end: u64,
    buffered: usize,
    at: usize,
    previous: Option<DirectoryKey>,
}
impl RunCursor {
    fn new(start: u64, end: u64) -> Self {
        Self {
            next: start,
            end,
            buffered: 0,
            at: 0,
            previous: None,
        }
    }
    fn next<R: ReadAt + ?Sized>(
        &mut self,
        source: &R,
        buffer: &mut [u8],
    ) -> Result<Option<LeafEntry>> {
        if self.at == self.buffered {
            if self.next == self.end {
                return Ok(None);
            }
            let records = (self.end - self.next).min((buffer.len() / RECORD_BYTES) as u64) as usize;
            let bytes = records * RECORD_BYTES;
            read_exact(
                source,
                self.next * RECORD_BYTES as u64,
                &mut buffer[..bytes],
            )?;
            self.next += records as u64;
            self.buffered = bytes;
            self.at = 0;
        }
        let entry = decode_record(&buffer[self.at..self.at + RECORD_BYTES])?;
        if self
            .previous
            .is_some_and(|key| key.identity() >= entry.key.identity())
        {
            return Err(RunError::KeyOrder);
        }
        self.at += RECORD_BYTES;
        self.previous = Some(entry.key);
        Ok(Some(entry))
    }
}

struct RecordOutput<'a, W: Write + ?Sized> {
    sink: &'a mut W,
    buffer: &'a mut [u8],
    len: usize,
}
impl<'a, W: Write + ?Sized> RecordOutput<'a, W> {
    fn new(sink: &'a mut W, buffer: &'a mut [u8]) -> Self {
        Self {
            sink,
            buffer,
            len: 0,
        }
    }
    fn push(&mut self, entry: LeafEntry) -> Result<()> {
        if self.buffer.len() - self.len < RECORD_BYTES {
            self.finish()?;
        }
        encode_record(entry, &mut self.buffer[self.len..self.len + RECORD_BYTES])?;
        self.len += RECORD_BYTES;
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        if self.len != 0 {
            self.sink.write_all(&self.buffer[..self.len])?;
            self.len = 0;
        }
        Ok(())
    }
}

fn encode_record(entry: LeafEntry, bytes: &mut [u8]) -> Result<()> {
    entry.key.validate()?;
    let descriptor = entry.page.encode()?;
    bytes[..2].copy_from_slice(&entry.key.section.to_le_bytes());
    bytes[2..4].copy_from_slice(&entry.key.flags.to_le_bytes());
    bytes[4..8].copy_from_slice(&entry.key.column.to_le_bytes());
    bytes[8..16].copy_from_slice(&entry.key.ordinal.to_le_bytes());
    bytes[16..LEAF_ENTRY_BYTES].copy_from_slice(&descriptor);
    let checksum = crc32fast::hash(&bytes[..LEAF_ENTRY_BYTES]);
    bytes[LEAF_ENTRY_BYTES..RECORD_BYTES].copy_from_slice(&checksum.to_le_bytes());
    Ok(())
}
fn decode_record(bytes: &[u8]) -> Result<LeafEntry> {
    let checksum = u32::from_le_bytes(bytes[LEAF_ENTRY_BYTES..].try_into().unwrap());
    if crc32fast::hash(&bytes[..LEAF_ENTRY_BYTES]) != checksum {
        return Err(RunError::Checksum);
    }
    Ok(LeafEntry::decode(&bytes[..LEAF_ENTRY_BYTES])?)
}
fn read_exact<R: ReadAt + ?Sized>(source: &R, mut offset: u64, mut bytes: &mut [u8]) -> Result<()> {
    while !bytes.is_empty() {
        match source.read_at(offset, bytes) {
            Ok(0) => return Err(RunError::UnexpectedEof),
            Ok(n) if n > bytes.len() => return Err(RunError::InvalidReadCount),
            Ok(n) => {
                offset += n as u64;
                bytes = &mut bytes[n..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => (),
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::directory::{Section, KEY_OPTIONAL, KEY_REQUIRED};
    use super::super::envelope::{Codec, PageDescriptor};
    use super::*;
    use std::cell::Cell;
    use std::io::Cursor;

    struct Stream {
        bytes: Cursor<Vec<u8>>,
        reads: Cell<usize>,
        read_limit: usize,
        interrupt: Cell<bool>,
    }
    impl Stream {
        fn new() -> Self {
            Self {
                bytes: Cursor::new(Vec::new()),
                reads: Cell::new(0),
                read_limit: usize::MAX,
                interrupt: Cell::new(false),
            }
        }
    }
    impl Write for Stream {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes.write(bytes)
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl Seek for Stream {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.bytes.seek(pos)
        }
    }
    impl ReadAt for Stream {
        fn read_at(&self, offset: u64, output: &mut [u8]) -> io::Result<usize> {
            self.reads.set(self.reads.get() + 1);
            if self.interrupt.replace(false) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let start = usize::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?;
            let source = self.bytes.get_ref().get(start..).unwrap_or_default();
            let n = source.len().min(output.len()).min(self.read_limit);
            output[..n].copy_from_slice(&source[..n]);
            Ok(n)
        }
    }
    fn entry(n: u64) -> LeafEntry {
        LeafEntry {
            key: DirectoryKey {
                section: Section::ColumnBlocks as u16,
                flags: KEY_REQUIRED,
                column: (n % 5) as u32,
                ordinal: n / 5,
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
    fn initial(source: &mut Stream, count: u64, capacity: usize) -> DescriptorRuns {
        let mut entries = vec![entry(0); capacity];
        let mut output = [0; IO_BYTES + 3];
        let mut writer = RunWriter::new(source, &mut entries, &mut output).unwrap();
        for n in (0..count).rev() {
            writer.push(entry(n)).unwrap();
        }
        let runs = writer.finish().unwrap();
        assert!(matches!(writer.finish(), Err(RunError::Finished)));
        runs
    }

    #[test]
    fn bounded_passes_preserve_every_descriptor_and_ignore_stale_tails() {
        for count in [0, 1, 2, 63, 64, 65, 127, 128, 129, 4097, 10001] {
            for capacity in [1, 17, 64, 65] {
                let mut a = Stream::new();
                let mut b = Stream::new();
                b.bytes = Cursor::new(vec![0xee; (count as usize + 7) * RECORD_BYTES]);
                let mut runs = initial(&mut a, count, capacity);
                let mut scratch = MergeScratch::new();
                while !runs.is_sorted() {
                    runs = merge_pass(&a, &mut b, runs, &mut scratch).unwrap();
                    std::mem::swap(&mut a, &mut b);
                    assert_eq!(runs.byte_len(), count * RECORD_BYTES as u64);
                }
                let mut expected: Vec<_> = (0..count).map(entry).collect();
                expected.sort_unstable_by(|a, b| a.key.cmp_identity(&b.key));
                let mut buffer = [0; IO_BYTES + 7];
                let mut reader = SortedReader::new(&a, runs, &mut buffer).unwrap();
                assert!(!reader.is_complete());
                for expected in expected {
                    assert_eq!(reader.next_entry().unwrap(), Some(expected));
                }
                assert_eq!(reader.next_entry().unwrap(), None);
                assert!(reader.is_complete());
                let reads = a.reads.get();
                assert_eq!(reader.next_entry().unwrap(), None);
                assert_eq!(a.reads.get(), reads);
            }
        }
    }

    #[test]
    fn checksum_short_reads_duplicates_and_preflight() {
        let mut a = Stream::new();
        let runs = initial(&mut a, 65, 64);
        let mut b = Stream::new();
        let mut scratch = MergeScratch::new();
        a.read_limit = 7;
        a.interrupt.set(true);
        let sorted = merge_pass(&a, &mut b, runs, &mut scratch).unwrap();
        assert!(sorted.is_sorted());
        let mut buffer = [0; RECORD_BYTES];
        assert!(matches!(
            SortedReader::new(&a, runs, &mut buffer),
            Err(RunError::NotSorted)
        ));
        let mut encoded = [0; RECORD_BYTES];
        encode_record(entry(7), &mut encoded).unwrap();
        assert_eq!(decode_record(&encoded).unwrap(), entry(7));
        for byte in 0..RECORD_BYTES {
            for bit in 0..8 {
                encoded[byte] ^= 1 << bit;
                assert!(matches!(decode_record(&encoded), Err(RunError::Checksum)));
                encoded[byte] ^= 1 << bit;
            }
        }
        let mut entries = [entry(0); 2];
        let mut output = [0; RECORD_BYTES];
        let mut writer = RunWriter::new(&mut a, &mut entries, &mut output).unwrap();
        let mut invalid = entry(0);
        invalid.key.flags = 0;
        assert!(matches!(writer.push(invalid), Err(RunError::Directory(_))));
        writer.push(entry(0)).unwrap();
        let mut duplicate = entry(0);
        duplicate.key.flags = KEY_OPTIONAL;
        assert!(matches!(writer.push(duplicate), Err(RunError::KeyOrder)));
        assert!(matches!(writer.finish(), Err(RunError::Poisoned)));

        // Duplicates in separate runs are discovered at their first merge.
        let mut one = [entry(0)];
        let mut writer = RunWriter::new(&mut a, &mut one, &mut output).unwrap();
        writer.push(entry(0)).unwrap();
        writer.push(duplicate).unwrap();
        let runs = writer.finish().unwrap();
        assert!(matches!(
            merge_pass(&a, &mut b, runs, &mut scratch),
            Err(RunError::KeyOrder)
        ));

        let runs = initial(&mut a, 1, 1);
        a.bytes.get_mut()[0] ^= 1;
        let mut reader = SortedReader::new(&a, runs, &mut buffer).unwrap();
        assert!(matches!(reader.next_entry(), Err(RunError::Checksum)));
        let reads = a.reads.get();
        assert!(matches!(reader.next_entry(), Err(RunError::Poisoned)));
        assert_eq!(a.reads.get(), reads);
        assert!(!reader.is_complete());
    }

    struct FaultSink {
        stream: Stream,
        left: usize,
        panic: bool,
    }
    impl Write for FaultSink {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.left == 0 {
                assert!(!self.panic, "injected metadata write panic");
                return Err(io::ErrorKind::WriteZero.into());
            }
            let n = bytes.len().min(self.left);
            self.left -= n;
            self.stream.write(&bytes[..n])
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    impl Seek for FaultSink {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            self.stream.seek(pos)
        }
    }
    #[test]
    fn partial_io_and_unwind_poison_the_accepted_run() {
        for panic in [false, true] {
            let mut sink = FaultSink {
                stream: Stream::new(),
                left: 7,
                panic,
            };
            let mut entries = [entry(0)];
            let mut output = [0; RECORD_BYTES];
            let mut writer = RunWriter::new(&mut sink, &mut entries, &mut output).unwrap();
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| writer.push(entry(0))));
            if panic {
                assert!(result.is_err());
            } else {
                assert!(matches!(result.unwrap(), Err(RunError::Io(_))));
            }
            assert!(matches!(writer.push(entry(1)), Err(RunError::Poisoned)));
            assert!(matches!(writer.finish(), Err(RunError::Poisoned)));
        }
        let mut source = Stream::new();
        let runs = initial(&mut source, 1, 1);
        source.bytes.get_mut().truncate(RECORD_BYTES - 1);
        let mut buffer = [0; RECORD_BYTES];
        let mut reader = SortedReader::new(&source, runs, &mut buffer).unwrap();
        assert!(matches!(reader.next_entry(), Err(RunError::UnexpectedEof)));
        assert!(matches!(reader.next_entry(), Err(RunError::Poisoned)));
    }

    #[test]
    fn merge_reads_whole_batches_and_bounded_logical_extents() {
        let mut a = Stream::new();
        let runs = initial(&mut a, 256, 128);
        let mut b = Stream::new();
        let sorted = merge_pass(&a, &mut b, runs, &mut MergeScratch::new()).unwrap();
        assert_eq!(a.reads.get(), 4);
        assert_eq!(b.bytes.get_ref().len() as u64, sorted.byte_len());
        let mut buffer = [0; IO_BYTES];
        let mut reader = SortedReader::new(&b, sorted, &mut buffer).unwrap();
        while reader.next_entry().unwrap().is_some() {}
        assert_eq!(b.reads.get(), 4);
        assert!(matches!(
            merge_pass(&b, &mut a, sorted, &mut MergeScratch::new()),
            Err(RunError::AlreadySorted)
        ));
    }
}
