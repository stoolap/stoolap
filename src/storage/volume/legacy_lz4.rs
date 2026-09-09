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

//! Bounded raw LZ4 block decoder for future V4 compatibility adapters.
//!
//! This module is not connected to volume reads yet. The caller supplies the
//! exact stored/decoded lengths and scratch, and must treat all output as
//! provisional until decoding and its outer checksum validation succeed.
//! There is no size prefix, LZ4 frame, external dictionary or durability step.
//! Successful decoding consumes exactly the declared block without prefetching
//! the following block. It does not call `Write::flush`.
//!
//! The grammar matches the locked lz4_flex 0.13 block decoder: high token nibble
//! gives literal length, low nibble plus four gives match length, and a nibble
//! of fifteen is extended by bytes through the first byte below 255. A final
//! literal-only sequence ends the block; an empty block is encoded as `[0]`.

use std::fmt;
use std::io::{self, Read, Write};

pub const HISTORY_BYTES: usize = 65_536;
pub const MAX_SCRATCH_BYTES: usize = 65_536;
const HISTORY_MASK: u64 = (HISTORY_BYTES - 1) as u64;

type Result<T> = std::result::Result<T, DecodeError>;

/// Format and state errors allocate nothing. The original I/O error is retained.
#[derive(Debug)]
pub enum DecodeError {
    InvalidScratch,
    Aborted,
    AlreadyFinished,
    TruncatedInput,
    LengthOverflow,
    LiteralOutOfBounds,
    OutputTooLong,
    DecodedLengthMismatch { expected: u64, actual: u64 },
    OffsetZero,
    OffsetOutOfBounds,
    InvalidReadCount,
    InvalidWriteCount,
    Io(io::Error),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidScratch => formatter.write_str("LZ4 scratch must contain 1..=65536 bytes"),
            Self::Aborted => formatter.write_str("LZ4 decoder is aborted"),
            Self::AlreadyFinished => formatter.write_str("LZ4 decoder already finished its block"),
            Self::TruncatedInput => formatter.write_str("truncated LZ4 block"),
            Self::LengthOverflow => formatter.write_str("LZ4 length overflow"),
            Self::LiteralOutOfBounds => {
                formatter.write_str("LZ4 literals exceed stored block length")
            }
            Self::OutputTooLong => {
                formatter.write_str("LZ4 output exceeds declared decoded length")
            }
            Self::DecodedLengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "LZ4 decoded length {actual} differs from declared {expected}"
                )
            }
            Self::OffsetZero => formatter.write_str("LZ4 match offset is zero"),
            Self::OffsetOutOfBounds => {
                formatter.write_str("LZ4 match offset precedes decoded history")
            }
            Self::InvalidReadCount => {
                formatter.write_str("LZ4 reader exceeded requested byte count")
            }
            Self::InvalidWriteCount => {
                formatter.write_str("LZ4 writer exceeded requested byte count")
            }
            Self::Io(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for DecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for DecodeError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeState {
    Ready,
    Aborted,
    Finished,
}

/// Length proof for this raw block, not checksum validation or durability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodedBlock {
    pub stored_bytes: u64,
    pub decoded_bytes: u64,
}

/// Single-block decoder. Reconstructing it explicitly reuses caller scratch
/// for a new block; there is no reset/resume operation after a partial failure.
/// The history is not cleared: offset validation prevents reading old contents
/// before the current block has generated the corresponding bytes.
///
/// Borrowed scratch totals at most 192 KiB; decoding itself allocates nothing.
/// Source/sink backing and the small decoder object are separate from scratch.
pub struct RawLz4Decoder<'a> {
    history: &'a mut [u8; HISTORY_BYTES],
    input_scratch: &'a mut [u8],
    output_scratch: &'a mut [u8],
    state: DecodeState,
    input_pos: usize,
    input_end: usize,
    output_used: usize,
    stored_len: u64,
    decoded_len: u64,
    stored_read: u64,
    stored_consumed: u64,
    produced: u64,
}

impl<'a> RawLz4Decoder<'a> {
    pub fn new(
        history: &'a mut [u8; HISTORY_BYTES],
        input_scratch: &'a mut [u8],
        output_scratch: &'a mut [u8],
    ) -> Result<Self> {
        if input_scratch.is_empty()
            || output_scratch.is_empty()
            || input_scratch.len() > MAX_SCRATCH_BYTES
            || output_scratch.len() > MAX_SCRATCH_BYTES
        {
            return Err(DecodeError::InvalidScratch);
        }
        Ok(Self {
            history,
            input_scratch,
            output_scratch,
            state: DecodeState::Ready,
            input_pos: 0,
            input_end: 0,
            output_used: 0,
            stored_len: 0,
            decoded_len: 0,
            stored_read: 0,
            stored_consumed: 0,
            produced: 0,
        })
    }

    pub fn state(&self) -> DecodeState {
        self.state
    }

    pub fn decode<R: Read, W: Write>(
        &mut self,
        input: &mut R,
        output: &mut W,
        stored_len: u64,
        decoded_len: u64,
    ) -> Result<DecodedBlock> {
        match self.state {
            DecodeState::Ready => {}
            DecodeState::Aborted => return Err(DecodeError::Aborted),
            DecodeState::Finished => return Err(DecodeError::AlreadyFinished),
        }
        // Poison before any arbitrary I/O, also covering unwinding callbacks.
        self.state = DecodeState::Aborted;
        self.stored_len = stored_len;
        self.decoded_len = decoded_len;
        self.decode_block(input, output)?;
        self.state = DecodeState::Finished;
        Ok(DecodedBlock {
            stored_bytes: self.stored_consumed,
            decoded_bytes: self.produced,
        })
    }

    fn decode_block<R: Read, W: Write>(&mut self, input: &mut R, output: &mut W) -> Result<()> {
        loop {
            let token = self.byte(input)?;
            let literal_length = self.length(input, u64::from(token >> 4), 15)?;
            if literal_length > self.stored_len - self.stored_consumed {
                return Err(DecodeError::LiteralOutOfBounds);
            }
            self.check_output(literal_length)?;
            self.literals(input, output, literal_length)?;
            if self.stored_consumed == self.stored_len {
                if self.produced != self.decoded_len {
                    return Err(DecodeError::DecodedLengthMismatch {
                        expected: self.decoded_len,
                        actual: self.produced,
                    });
                }
                self.drain_output(output)?;
                return Ok(());
            }
            let offset = usize::from(u16::from_le_bytes([self.byte(input)?, self.byte(input)?]));
            if offset == 0 {
                return Err(DecodeError::OffsetZero);
            }
            if offset as u64 > self.produced {
                return Err(DecodeError::OffsetOutOfBounds);
            }
            let match_length = self.length(input, 4 + u64::from(token & 15), 19)?;
            self.check_output(match_length)?;
            self.matches(output, offset, match_length)?;
            // A match cannot terminate the block. The next sequence must
            // provide a token, including the zero token for no final literals.
        }
    }

    fn check_output(&self, length: u64) -> Result<()> {
        if length > self.decoded_len - self.produced {
            Err(DecodeError::OutputTooLong)
        } else {
            Ok(())
        }
    }

    fn length<R: Read>(&mut self, input: &mut R, initial: u64, extended: u64) -> Result<u64> {
        let mut length = initial;
        if initial == extended {
            loop {
                let extra = self.byte(input)?;
                length = length
                    .checked_add(u64::from(extra))
                    .ok_or(DecodeError::LengthOverflow)?;
                // No need to read an unbounded extension after output already
                // exceeds the caller's declared limit.
                self.check_output(length)?;
                if extra != 255 {
                    break;
                }
            }
        }
        Ok(length)
    }

    fn refill<R: Read>(&mut self, input: &mut R) -> Result<()> {
        if self.input_pos < self.input_end {
            return Ok(());
        }
        let remaining = self.stored_len - self.stored_read;
        if remaining == 0 {
            return Err(DecodeError::TruncatedInput);
        }
        let requested = remaining.min(self.input_scratch.len() as u64) as usize;
        let read = loop {
            match input.read(&mut self.input_scratch[..requested]) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
                Ok(0) => return Err(DecodeError::TruncatedInput),
                Ok(length) if length > requested => return Err(DecodeError::InvalidReadCount),
                Ok(length) => break length,
            }
        };
        self.stored_read += read as u64;
        self.input_pos = 0;
        self.input_end = read;
        Ok(())
    }

    #[inline]
    fn byte<R: Read>(&mut self, input: &mut R) -> Result<u8> {
        self.refill(input)?;
        let byte = self.input_scratch[self.input_pos];
        self.input_pos += 1;
        self.stored_consumed += 1;
        Ok(byte)
    }

    fn literals<R: Read, W: Write>(
        &mut self,
        input: &mut R,
        output: &mut W,
        mut remaining: u64,
    ) -> Result<()> {
        while remaining != 0 {
            self.refill(input)?;
            let count = remaining
                .min((self.input_end - self.input_pos) as u64)
                .min((self.output_scratch.len() - self.output_used) as u64)
                as usize;
            self.output_scratch[self.output_used..self.output_used + count]
                .copy_from_slice(&self.input_scratch[self.input_pos..self.input_pos + count]);
            self.input_pos += count;
            self.stored_consumed += count as u64;
            self.record_output(count);
            remaining -= count as u64;
            if self.output_used == self.output_scratch.len() {
                self.drain_output(output)?;
            }
        }
        Ok(())
    }

    fn matches<W: Write>(
        &mut self,
        output: &mut W,
        offset: usize,
        mut remaining: u64,
    ) -> Result<()> {
        while remaining != 0 {
            let count =
                remaining.min((self.output_scratch.len() - self.output_used) as u64) as usize;
            let start = self.output_used;
            let seed = count.min(offset);
            let source = ((self.produced - offset as u64) & HISTORY_MASK) as usize;
            let first = seed.min(HISTORY_BYTES - source);
            self.output_scratch[start..start + first]
                .copy_from_slice(&self.history[source..source + first]);
            self.output_scratch[start + first..start + seed]
                .copy_from_slice(&self.history[..seed - first]);
            // One period is sufficient, even for offset one. Copy already
            // generated output in doubling spans instead of a bytewise loop.
            let mut copied = seed;
            while copied < count {
                let next = copied.min(count - copied);
                self.output_scratch
                    .copy_within(start..start + next, start + copied);
                copied += next;
            }
            self.record_output(count);
            remaining -= count as u64;
            if self.output_used == self.output_scratch.len() {
                self.drain_output(output)?;
            }
        }
        Ok(())
    }

    fn record_output(&mut self, count: usize) {
        let position = (self.produced & HISTORY_MASK) as usize;
        let first = count.min(HISTORY_BYTES - position);
        let bytes = &self.output_scratch[self.output_used..self.output_used + count];
        self.history[position..position + first].copy_from_slice(&bytes[..first]);
        self.history[..count - first].copy_from_slice(&bytes[first..]);
        self.output_used += count;
        self.produced += count as u64;
    }

    fn drain_output<W: Write>(&mut self, output: &mut W) -> Result<()> {
        if self.output_used != 0 {
            let mut written = 0;
            while written < self.output_used {
                let pending = &self.output_scratch[written..self.output_used];
                match output.write(pending) {
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error.into()),
                    Ok(0) => return Err(io::Error::from(io::ErrorKind::WriteZero).into()),
                    Ok(length) if length > pending.len() => {
                        return Err(DecodeError::InvalidWriteCount)
                    }
                    Ok(length) => written += length,
                }
            }
            self.output_used = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn decode(
        block: &[u8],
        expected: usize,
        input_size: usize,
        output_size: usize,
    ) -> Result<Vec<u8>> {
        let mut history = [0xa5; HISTORY_BYTES];
        let mut input_scratch = vec![0; input_size];
        let mut output_scratch = vec![0; output_size];
        let mut decoder =
            RawLz4Decoder::new(&mut history, &mut input_scratch, &mut output_scratch)?;
        let mut output = Vec::new();
        let receipt = decoder.decode(
            &mut &*block,
            &mut output,
            block.len() as u64,
            expected as u64,
        )?;
        assert_eq!(
            receipt,
            DecodedBlock {
                stored_bytes: block.len() as u64,
                decoded_bytes: expected as u64
            }
        );
        assert_eq!(decoder.state(), DecodeState::Finished);
        Ok(output)
    }

    fn extension(block: &mut Vec<u8>, mut extra: usize) {
        while extra >= 255 {
            block.push(255);
            extra -= 255;
        }
        block.push(extra as u8);
    }

    fn sequence(block: &mut Vec<u8>, literals: &[u8], offset: u16, match_length: usize) {
        assert!(match_length >= 4);
        block.push(((literals.len().min(15) as u8) << 4) | ((match_length - 4).min(15) as u8));
        if literals.len() >= 15 {
            extension(block, literals.len() - 15);
        }
        block.extend_from_slice(literals);
        block.extend_from_slice(&offset.to_le_bytes());
        if match_length >= 19 {
            extension(block, match_length - 19);
        }
    }

    #[test]
    fn valid_blocks_match_lz4_flex_across_lengths_and_small_scratch() {
        let mut seed = 0x1234_5678u32;
        for length in [
            0, 1, 4, 14, 15, 16, 19, 254, 255, 270, 4096, 65_535, 65_536, 65_537, 131_111,
        ] {
            for pattern in 0..4 {
                let input: Vec<_> = (0..length)
                    .map(|index| match pattern {
                        0 => 42,
                        1 => (index % 251) as u8,
                        2 => {
                            seed ^= seed << 13;
                            seed ^= seed >> 17;
                            seed ^= seed << 5;
                            seed as u8
                        }
                        _ => ((index / 8191 + index % 17) % 255) as u8,
                    })
                    .collect();
                let block = lz4_flex::block::compress(&input);
                assert_eq!(lz4_flex::block::decompress(&block, length).unwrap(), input);
                for (in_size, out_size) in [(1, 1), (7, 13), (4096, 8192)] {
                    assert_eq!(
                        decode(&block, length, in_size, out_size).unwrap(),
                        input,
                        "length={length}, pattern={pattern}, scratch={in_size}/{out_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn overlapping_matches_and_ring_wrap_preserve_back_reference_age() {
        let prefix: Vec<_> = (0..65_617)
            .map(|n| ((n * 17 + n / 251) % 253) as u8)
            .collect();
        for offset in [1u16, 2, 3, 4, 7, 8, 15, 16, 31, 255, 4095, 32768, 65535] {
            for length in [4, usize::from(offset).max(4), usize::from(offset) + 65_559] {
                let mut block = Vec::new();
                sequence(&mut block, &prefix, offset, length);
                // Match source and destination wrap differently when offset
                // does not divide either the ring or output scratch length.
                sequence(&mut block, b"abc", offset, 71_003);
                block.extend_from_slice(&[0x50, b'e', b'n', b'd', b'!', b'!']);
                let expected_len = prefix.len() + length + 3 + 71_003 + 5;
                let reference = lz4_flex::block::decompress(&block, expected_len).unwrap();
                for (in_size, out_size) in [(37, 73), (4096, 4093), (65536, 65536)] {
                    assert_eq!(
                        decode(&block, expected_len, in_size, out_size).unwrap(),
                        reference,
                        "offset={offset}, length={length}, output={out_size}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_block_and_stored_boundary_match_reference_without_prefetch() {
        assert!(lz4_flex::block::decompress(&[], 0).is_err());
        assert!(matches!(
            decode(&[], 0, 1, 1),
            Err(DecodeError::TruncatedInput)
        ));
        assert_eq!(
            lz4_flex::block::decompress(&[0], 0).unwrap(),
            Vec::<u8>::new()
        );
        assert!(decode(&[0], 0, 1, 1).unwrap().is_empty());
        // lz4_flex ignores the match nibble in the last literal-only sequence.
        assert!(decode(&[15], 0, 1, 1).unwrap().is_empty());
        let input = b"a block whose following bytes are owned by the next block";
        let block = lz4_flex::block::compress(input);
        let stored = block.len();
        let mut containing = block;
        containing.extend_from_slice(b"NEXT BLOCK");
        let mut cursor = Cursor::new(containing);
        let mut history = [0; HISTORY_BYTES];
        let mut scratch_in = [0; 65536];
        let mut scratch_out = [0; 31];
        let mut decoder =
            RawLz4Decoder::new(&mut history, &mut scratch_in, &mut scratch_out).unwrap();
        let mut output = Vec::new();
        decoder
            .decode(&mut cursor, &mut output, stored as u64, input.len() as u64)
            .unwrap();
        assert_eq!(cursor.position(), stored as u64);
        assert_eq!(output, input);
        let position = cursor.position();
        assert!(matches!(
            decoder.decode(&mut cursor, &mut output, 0, 0),
            Err(DecodeError::AlreadyFinished)
        ));
        assert_eq!(cursor.position(), position);
    }

    #[test]
    fn malformed_tokens_offsets_and_exact_lengths_fail_closed() {
        let malformed: &[&[u8]] = &[
            &[],
            &[0xff],
            &[0xf0, 255],
            &[0x10],
            &[0x10, b'a', 0],
            &[0x10, b'a', 0, 0, 0],
            &[0x00, 1, 0, 0],
            &[0x10, b'a', 2, 0, 0],
            &[0x1f, b'a', 1, 0, 255],
            &[0x10, b'a', 1, 0],
        ];
        for block in malformed {
            assert!(
                lz4_flex::block::decompress(block, 1024).is_err(),
                "reference {block:?}"
            );
            assert!(decode(block, 1024, 3, 7).is_err(), "stream {block:?}");
        }
        assert!(matches!(
            decode(&[0x10, b'a', 0, 0, 0], 5, 1, 1),
            Err(DecodeError::OffsetZero)
        ));
        assert!(matches!(
            decode(&[0x10, b'a', 2, 0, 0], 5, 1, 1),
            Err(DecodeError::OffsetOutOfBounds)
        ));
        let original = b"abcdefghabcdefghabcdefghabcdefghabcdefgh0123456789";
        let block = lz4_flex::block::compress(original);
        for end in 0..block.len() {
            assert!(
                decode(&block[..end], original.len(), 5, 11).is_err(),
                "end={end}"
            );
        }
        assert!(matches!(
            decode(&block, original.len() - 1, 5, 11),
            Err(DecodeError::OutputTooLong)
        ));
        assert!(matches!(
            decode(&block, original.len() + 1, 5, 11),
            Err(DecodeError::DecodedLengthMismatch { .. })
        ));
        // Extended length arithmetic is checked even at the u64 boundary.
        let mut history = [0; HISTORY_BYTES];
        let mut scratch_in = [0; 1];
        let mut scratch_out = [0; 1];
        let mut decoder =
            RawLz4Decoder::new(&mut history, &mut scratch_in, &mut scratch_out).unwrap();
        decoder.stored_len = 1;
        decoder.decoded_len = u64::MAX;
        assert!(matches!(
            decoder.length(&mut &[1u8][..], u64::MAX, u64::MAX),
            Err(DecodeError::LengthOverflow)
        ));
    }

    struct ShortReader<'a> {
        bytes: &'a [u8],
        calls: usize,
        fail_after: Option<usize>,
        read: usize,
    }
    impl Read for ShortReader<'_> {
        fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.fail_after == Some(self.read) {
                return Err(io::Error::other("injected read error"));
            }
            if self.calls % 3 == 0 {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let length = output
                .len()
                .min(3)
                .min(self.bytes.len())
                .min(self.fail_after.map_or(usize::MAX, |end| end - self.read));
            output[..length].copy_from_slice(&self.bytes[..length]);
            self.bytes = &self.bytes[length..];
            self.read += length;
            Ok(length)
        }
    }
    struct ShortWriter {
        bytes: Vec<u8>,
        calls: usize,
        fail_after: Option<usize>,
    }
    impl Write for ShortWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            if self.fail_after == Some(self.bytes.len()) {
                return Err(io::Error::other("injected write error"));
            }
            if self.calls % 3 == 0 {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let length = bytes.len().min(5).min(
                self.fail_after
                    .map_or(usize::MAX, |end| end - self.bytes.len()),
            );
            self.bytes.extend_from_slice(&bytes[..length]);
            Ok(length)
        }
        fn flush(&mut self) -> io::Result<()> {
            panic!("raw block decoder must not flush")
        }
    }

    #[test]
    fn short_interrupted_io_and_partial_failure_are_sticky() {
        let original = b"hello hello hello hello hello hello hello!";
        let block = lz4_flex::block::compress(original);
        for fail_read in std::iter::once(None).chain((0..block.len()).map(Some)) {
            let mut history = [0; HISTORY_BYTES];
            let mut scratch_in = [0; 11];
            let mut scratch_out = [0; 13];
            let mut decoder =
                RawLz4Decoder::new(&mut history, &mut scratch_in, &mut scratch_out).unwrap();
            let mut reader = ShortReader {
                bytes: &block,
                calls: 0,
                fail_after: fail_read,
                read: 0,
            };
            let mut writer = ShortWriter {
                bytes: Vec::new(),
                calls: 0,
                fail_after: None,
            };
            let result = decoder.decode(
                &mut reader,
                &mut writer,
                block.len() as u64,
                original.len() as u64,
            );
            if fail_read.is_none() {
                result.unwrap();
                assert_eq!(writer.bytes, original);
            } else {
                assert!(matches!(result, Err(DecodeError::Io(_))));
                assert_eq!(decoder.state(), DecodeState::Aborted);
                let calls = (reader.calls, writer.calls);
                assert!(matches!(
                    decoder.decode(&mut reader, &mut writer, 1, 0),
                    Err(DecodeError::Aborted)
                ));
                assert_eq!((reader.calls, writer.calls), calls);
            }
        }
        for fail_at in 0..original.len() {
            let mut history = [0; HISTORY_BYTES];
            let mut scratch_in = [0; 11];
            let mut scratch_out = [0; 13];
            let mut decoder =
                RawLz4Decoder::new(&mut history, &mut scratch_in, &mut scratch_out).unwrap();
            let mut reader = ShortReader {
                bytes: &block,
                calls: 0,
                fail_after: None,
                read: 0,
            };
            let mut writer = ShortWriter {
                bytes: Vec::new(),
                calls: 0,
                fail_after: Some(fail_at),
            };
            assert!(matches!(
                decoder.decode(
                    &mut reader,
                    &mut writer,
                    block.len() as u64,
                    original.len() as u64
                ),
                Err(DecodeError::Io(_))
            ));
            assert_eq!(writer.bytes, &original[..fail_at]);
            let calls = (reader.calls, writer.calls);
            assert!(matches!(
                decoder.decode(&mut reader, &mut writer, 1, 0),
                Err(DecodeError::Aborted)
            ));
            assert_eq!((reader.calls, writer.calls), calls);
        }
    }

    #[test]
    fn invalid_io_counts_and_write_zero_return_errors_without_resuming() {
        struct LyingReader(usize);
        impl Read for LyingReader {
            fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
                self.0 += 1;
                Ok(bytes.len() + 1)
            }
        }
        struct LyingWriter {
            calls: usize,
            zero: bool,
        }
        impl Write for LyingWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                self.calls += 1;
                Ok(if self.zero { 0 } else { bytes.len() + 1 })
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("no flush")
            }
        }
        let mut history = [0; HISTORY_BYTES];
        let mut scratch_in = [0; 8];
        let mut scratch_out = [0; 8];
        let mut reader = LyingReader(0);
        let mut output = Vec::new();
        let mut decoder =
            RawLz4Decoder::new(&mut history, &mut scratch_in, &mut scratch_out).unwrap();
        assert!(matches!(
            decoder.decode(&mut reader, &mut output, 2, 1),
            Err(DecodeError::InvalidReadCount)
        ));
        assert_eq!(reader.0, 1);
        assert!(output.is_empty());
        assert_eq!(decoder.stored_read, 0);
        assert_eq!(decoder.input_end, 0);
        assert!(matches!(
            decoder.decode(&mut reader, &mut output, 2, 1),
            Err(DecodeError::Aborted)
        ));
        assert_eq!(reader.0, 1);
        for zero in [false, true] {
            let mut decoder =
                RawLz4Decoder::new(&mut history, &mut scratch_in, &mut scratch_out).unwrap();
            let mut writer = LyingWriter { calls: 0, zero };
            let error = decoder
                .decode(&mut &[0x10, b'x'][..], &mut writer, 2, 1)
                .unwrap_err();
            if zero {
                assert!(
                    matches!(error, DecodeError::Io(ref error) if error.kind() == io::ErrorKind::WriteZero)
                );
            } else {
                assert!(matches!(error, DecodeError::InvalidWriteCount));
            }
            assert_eq!(decoder.state(), DecodeState::Aborted);
            assert!(matches!(
                decoder.decode(&mut reader, &mut writer, 2, 1),
                Err(DecodeError::Aborted)
            ));
            assert_eq!(reader.0, 1);
            assert_eq!(writer.calls, 1);
        }
    }

    #[test]
    fn panic_also_poisoned_before_first_io_and_scratch_is_bounded() {
        struct PanicIo;
        impl Read for PanicIo {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                panic!("reader panic")
            }
        }
        impl Write for PanicIo {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                panic!("writer panic")
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("flush panic")
            }
        }
        let mut history = [0; HISTORY_BYTES];
        let mut scratch_in = [0; 1];
        let mut scratch_out = [0; 1];
        for writing in [false, true] {
            let mut decoder =
                RawLz4Decoder::new(&mut history, &mut scratch_in, &mut scratch_out).unwrap();
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if writing {
                    decoder.decode(&mut &[0x10, b'x'][..], &mut PanicIo, 2, 1)
                } else {
                    decoder.decode(&mut PanicIo, &mut io::sink(), 1, 0)
                }
            }));
            assert!(panicked.is_err());
            assert_eq!(decoder.state(), DecodeState::Aborted);
            assert!(matches!(
                decoder.decode(&mut PanicIo, &mut PanicIo, 1, 0),
                Err(DecodeError::Aborted)
            ));
        }
        let mut empty = [];
        assert!(matches!(
            RawLz4Decoder::new(&mut history, &mut empty, &mut scratch_out),
            Err(DecodeError::InvalidScratch)
        ));
        assert!(matches!(
            RawLz4Decoder::new(&mut history, &mut scratch_in, &mut empty),
            Err(DecodeError::InvalidScratch)
        ));
        let mut too_large = [0; MAX_SCRATCH_BYTES + 1];
        assert!(matches!(
            RawLz4Decoder::new(&mut history, &mut too_large, &mut scratch_out),
            Err(DecodeError::InvalidScratch)
        ));
    }
}
