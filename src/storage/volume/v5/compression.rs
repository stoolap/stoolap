// Copyright 2026 Stoolap Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bounded page compression with caller-owned scratch and a reusable LZ4 table.
//! The plan exposes the output reservation before compression. Input, output
//! and the compression table coexist; the lifecycle budget must include all
//! three. Creating the table allocates outside this module (8/16 KiB of entries
//! for the locked lz4_flex version). A call never upgrades or allocates a table.
//!
//! Incompressible pages borrow the original raw input. A compressed page borrows
//! only its initialized output prefix. These bytes are not a volume or a
//! durability receipt; PageWriter still supplies framing and checksums.

use std::fmt;

use lz4_flex::block::{compress_into_with_table, get_maximum_output_size, CompressTable};

use super::column_block::MAX_DECODED_BYTES;
use super::envelope::{Codec, ReadLimits};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompressionError {
    Empty,
    DecodedLimit,
    InputLength,
    OutputTooShort,
    LargeTableRequired,
    StoredLimit,
    Codec,
}
impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V5 page compression: {self:?}")
    }
}
impl std::error::Error for CompressionError {}
type Result<T> = std::result::Result<T, CompressionError>;

#[derive(Clone, Copy, Debug)]
pub struct CompressionPlan {
    decoded_len: usize,
    stored_limit: u64,
}
impl CompressionPlan {
    pub fn new(decoded_len: usize, limits: &ReadLimits) -> Result<Self> {
        if decoded_len == 0 {
            return Err(CompressionError::Empty);
        }
        if decoded_len > MAX_DECODED_BYTES || decoded_len as u64 > limits.page_decoded_bytes {
            return Err(CompressionError::DecodedLimit);
        }
        if limits.page_stored_bytes == 0 {
            return Err(CompressionError::StoredLimit);
        }
        Ok(Self {
            decoded_len,
            stored_limit: limits.page_stored_bytes,
        })
    }
    /// Checked input length is at most 1 MiB, so the dependency's reservation
    /// formula cannot overflow either usize or its intermediate u64 arithmetic.
    pub fn output_capacity(self) -> usize {
        get_maximum_output_size(self.decoded_len)
    }
    /// Table size is a reservation decision; reject before touching scratch
    /// instead of allowing the dependency to transparently allocate an upgrade.
    pub fn requires_large_table(self) -> bool {
        self.decoded_len >= u16::MAX as usize
    }
    /// Length/table failures leave output untouched. StoredLimit can also occur
    /// after compression, when neither representation fits; output is reusable
    /// scratch and no external write has occurred.
    pub fn compress<'a>(
        self,
        decoded: &'a [u8],
        output: &'a mut [u8],
        table: &mut CompressTable,
    ) -> Result<StoredBlock<'a>> {
        if decoded.len() != self.decoded_len {
            return Err(CompressionError::InputLength);
        }
        if output.len() < self.output_capacity() {
            return Err(CompressionError::OutputTooShort);
        }
        if self.requires_large_table() && matches!(table, CompressTable::Small(_)) {
            return Err(CompressionError::LargeTableRequired);
        }
        let count = compress_into_with_table(decoded, &mut output[..self.output_capacity()], table)
            .map_err(|_| CompressionError::Codec)?;
        let (codec, bytes) = if count < decoded.len() && count as u64 <= self.stored_limit {
            (Codec::Lz4Block, &output[..count])
        } else if decoded.len() as u64 <= self.stored_limit {
            (Codec::Raw, decoded)
        } else {
            return Err(CompressionError::StoredLimit);
        };
        Ok(StoredBlock {
            codec,
            bytes,
            decoded_len: self.decoded_len,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StoredBlock<'a> {
    codec: Codec,
    bytes: &'a [u8],
    decoded_len: usize,
}
impl<'a> StoredBlock<'a> {
    pub const fn codec(self) -> Codec {
        self.codec
    }
    pub const fn bytes(self) -> &'a [u8] {
        self.bytes
    }
    pub const fn decoded_len(self) -> usize {
        self.decoded_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits() -> ReadLimits {
        ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: MAX_DECODED_BYTES as u64,
            page_decoded_bytes: MAX_DECODED_BYTES as u64,
        }
    }
    #[test]
    fn independent_blocks_reuse_tables_and_borrow_only_valid_bytes() {
        let mut table = CompressTable::large();
        let mut state = 123456789u64;
        for length in [1, 12, 64, 65_534, 65_535, MAX_DECODED_BYTES, 257] {
            for random in [false, true] {
                let input: Vec<u8> = (0..length)
                    .map(|i| {
                        if random {
                            state ^= state << 13;
                            state ^= state >> 7;
                            state ^= state << 17;
                            state as u8
                        } else {
                            (i % 7) as u8
                        }
                    })
                    .collect();
                let plan = CompressionPlan::new(length, &limits()).unwrap();
                let mut output = vec![0xa5; plan.output_capacity() + 8];
                let block = plan.compress(&input, &mut output, &mut table).unwrap();
                assert_eq!(block.decoded_len(), length);
                match block.codec() {
                    Codec::Raw => {
                        assert_eq!(block.bytes(), input);
                        assert_eq!(block.bytes().as_ptr(), input.as_ptr());
                    }
                    Codec::Lz4Block => {
                        assert!(block.bytes().len() < length);
                        let mut decoded = vec![0; length];
                        assert_eq!(
                            lz4_flex::block::decompress_into(block.bytes(), &mut decoded).unwrap(),
                            length
                        );
                        assert_eq!(decoded, input);
                    }
                }
                assert_eq!(output[plan.output_capacity()..], [0xa5; 8]);
            }
        }
    }
    #[test]
    fn reservations_and_stored_limits_fail_without_allocating_an_upgrade() {
        let mut small = CompressTable::small();
        let plan = CompressionPlan::new(65_535, &limits()).unwrap();
        let input = vec![0; 65_535];
        let mut output = vec![0xa5; plan.output_capacity()];
        assert_eq!(
            plan.compress(&input, &mut output, &mut small).unwrap_err(),
            CompressionError::LargeTableRequired
        );
        assert!(matches!(small, CompressTable::Small(_)));
        assert!(output.iter().all(|&v| v == 0xa5));
        assert_eq!(
            plan.compress(&input[..10], &mut output, &mut small)
                .unwrap_err(),
            CompressionError::InputLength
        );
        assert_eq!(
            plan.compress(&input, &mut output[..10], &mut small)
                .unwrap_err(),
            CompressionError::OutputTooShort
        );
        assert!(output.iter().all(|&v| v == 0xa5));
        let tiny = ReadLimits {
            page_stored_bytes: 1,
            ..limits()
        };
        let plan = CompressionPlan::new(1, &tiny).unwrap();
        assert_eq!(
            plan.compress(&[7], &mut output, &mut small)
                .unwrap()
                .codec(),
            Codec::Raw
        );
        let plan = CompressionPlan::new(1024, &tiny).unwrap();
        assert_eq!(
            plan.compress(&input[..1024], &mut output, &mut small)
                .unwrap_err(),
            CompressionError::StoredLimit
        );
        let fitting = ReadLimits {
            page_stored_bytes: 64,
            ..limits()
        };
        let plan = CompressionPlan::new(1024, &fitting).unwrap();
        assert_eq!(
            plan.compress(&input[..1024], &mut output, &mut small)
                .unwrap()
                .codec(),
            Codec::Lz4Block
        );
        assert_eq!(
            CompressionPlan::new(0, &limits()).unwrap_err(),
            CompressionError::Empty
        );
        assert_eq!(
            CompressionPlan::new(usize::MAX, &limits()).unwrap_err(),
            CompressionError::DecodedLimit
        );
        assert_eq!(
            CompressionPlan::new(
                2,
                &ReadLimits {
                    page_decoded_bytes: 1,
                    ..limits()
                }
            )
            .unwrap_err(),
            CompressionError::DecodedLimit
        );
    }
}
