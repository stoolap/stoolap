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

//! Allocation-free V5 envelope codecs. All numeric fields are little endian.
//!
//! Header (64 bytes):
//! ```text
//!  0..4   STV5                 4..6   envelope revision (1)
//!  6..8   header size (64)     8..16  required feature flags
//! 16..24  optional flags      24..32  nonzero TableId
//! 32..40  nonzero incarnation 40..48  nonzero volume ID
//! 48..60  reserved zero       60..64  CRC32 of bytes 0..60
//! ```
//! Footer (64 bytes):
//! ```text
//!  0..4   V5FT                 4..6   envelope revision (1)
//!  6..8   footer size (64)     8..16  exact file length
//! 16..48  root descriptor     48..60  reserved zero
//! 60..64  CRC32 of bytes 0..60
//! ```
//! Page descriptor (32 bytes, also embedded in independently checked directory
//! pages): `offset:u64, stored_len:u64, decoded_len:u64, stored_crc32:u32,
//! codec:u8, reserved_zero:[u8;3]`. Codec 0 is Raw; 1 is an LZ4 block. Equal
//! lengths never select a codec. Checksums cover stored bytes, not decoded data.
//!
//! The root is nonempty even for an empty volume and immediately precedes the
//! footer. All ordinary pages precede the root and follow the header. Directory
//! structure, page overlap/cycle detection, group ranges and payload validation
//! belong to the directory/payload readers, not these envelope-only codecs.
//!
//! Table IDs are never zero. Incarnations start at 1; TRUNCATE allocates a new
//! nonzero incarnation under the future durable catalog protocol. This module
//! accepts identities supplied by its caller; it invents none and enables no
//! production writes. Before allocating a read buffer, the caller must reserve
//! its memory and convert the validated u64 byte length to usize with checking.

use std::fmt;
use std::num::NonZeroU64;

pub const HEADER_SIZE: usize = 64;
pub const FOOTER_SIZE: usize = 64;
pub const DESCRIPTOR_SIZE: usize = 32;
pub const ENVELOPE_VERSION: u16 = 1;

/// Mandatory from the first V5 revision: rows carry paged source-DML LSNs.
pub const REQUIRED_SOURCE_LSNS: u64 = 1 << 0;
/// Mandatory from the first V5 revision: group ranges are explicit metadata.
pub const REQUIRED_GROUP_RANGES: u64 = 1 << 1;
pub const REQUIRED_FEATURES: u64 = REQUIRED_SOURCE_LSNS | REQUIRED_GROUP_RANGES;
/// Required only when the root carries LegacyBase(E,G): source lane zero then
/// names that checkpoint, never an absent/unlogged numeric LSN.
pub const REQUIRED_LEGACY_BASE: u64 = 1 << 2;
pub const KNOWN_REQUIRED_FEATURES: u64 = REQUIRED_FEATURES | REQUIRED_LEGACY_BASE;

/// Optional pruning sections. Neither flag proves a key's absence; each page
/// still needs its own supported-kind, checksum and payload validation.
pub const OPTIONAL_BLOOMS: u64 = 1 << 0;
pub const OPTIONAL_CONSTRAINT_SUMMARIES: u64 = 1 << 1;
pub const KNOWN_OPTIONAL_FEATURES: u64 = OPTIONAL_BLOOMS | OPTIONAL_CONSTRAINT_SUMMARIES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityField {
    Table,
    Incarnation,
    Volume,
}

/// A scalar error representation: malformed bytes do not allocate an error
/// string or a buffer sized from untrusted metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnvelopeError {
    InvalidLength { expected: usize, actual: usize },
    ChecksumMismatch,
    InvalidMagic,
    UnsupportedVersion(u16),
    InvalidDeclaredSize(u16),
    ReservedBytes,
    UnknownRequiredFeatures(u64),
    MissingRequiredFeatures(u64),
    ZeroIdentity(IdentityField),
    UnknownCodec(u8),
    EmptyPage,
    RawLengthMismatch,
    StoredLimitExceeded,
    DecodedLimitExceeded,
    OffsetOverflow,
    PageOutOfBounds,
    FileTooShort,
    FileLengthMismatch,
    StoredLengthMismatch,
    RootNotLast,
}

impl fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, actual } => {
                write!(f, "V5 envelope length {actual}, expected {expected}")
            }
            Self::ChecksumMismatch => f.write_str("V5 stored-byte checksum mismatch"),
            Self::InvalidMagic => f.write_str("invalid V5 envelope magic"),
            Self::UnsupportedVersion(version) => write!(f, "unsupported V5 revision {version}"),
            Self::InvalidDeclaredSize(size) => write!(f, "invalid V5 envelope size {size}"),
            Self::ReservedBytes => f.write_str("nonzero V5 reserved bytes"),
            Self::UnknownRequiredFeatures(bits) => {
                write!(f, "unsupported V5 required features {bits:#x}")
            }
            Self::MissingRequiredFeatures(bits) => {
                write!(f, "missing mandatory V5 features {bits:#x}")
            }
            Self::ZeroIdentity(field) => write!(f, "zero V5 {field:?} identity"),
            Self::UnknownCodec(codec) => write!(f, "unsupported V5 page codec {codec}"),
            Self::EmptyPage => f.write_str("empty V5 page"),
            Self::RawLengthMismatch => f.write_str("V5 raw page stored/decoded lengths differ"),
            Self::StoredLimitExceeded => f.write_str("V5 stored page exceeds byte limit"),
            Self::DecodedLimitExceeded => f.write_str("V5 decoded page exceeds byte limit"),
            Self::OffsetOverflow => f.write_str("V5 page offset overflow"),
            Self::PageOutOfBounds => f.write_str("V5 page lies outside its file region"),
            Self::FileTooShort => f.write_str("V5 file has no room for a nonempty root"),
            Self::FileLengthMismatch => {
                f.write_str("V5 footer file length differs from opened file")
            }
            Self::StoredLengthMismatch => {
                f.write_str("V5 stored bytes differ from descriptor length")
            }
            Self::RootNotLast => f.write_str("V5 root does not immediately precede footer"),
        }
    }
}
impl std::error::Error for EnvelopeError {}

type Result<T> = std::result::Result<T, EnvelopeError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FileIdentity {
    pub table_id: NonZeroU64,
    pub incarnation: NonZeroU64,
    pub volume_id: NonZeroU64,
}

/// Decoded metadata only, not evidence that the upgrade checkpoint is durable.
/// Phase5 must install the complete canonical recovered state as LegacyBase(E,G)
/// before publishing E; known pre-barrier row LSNs are not kept as competing
/// numeric sources in that state. Post-barrier DML sources must be greater than
/// G. Conversion preserves this pair and never invents a source from current LSN.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LegacyBase {
    pub generation: NonZeroU64,
    pub barrier_lsn: u64,
}
impl FileIdentity {
    pub fn new(table_id: u64, incarnation: u64, volume_id: u64) -> Result<Self> {
        Ok(Self {
            table_id: NonZeroU64::new(table_id)
                .ok_or(EnvelopeError::ZeroIdentity(IdentityField::Table))?,
            incarnation: NonZeroU64::new(incarnation)
                .ok_or(EnvelopeError::ZeroIdentity(IdentityField::Incarnation))?,
            volume_id: NonZeroU64::new(volume_id)
                .ok_or(EnvelopeError::ZeroIdentity(IdentityField::Volume))?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub identity: FileIdentity,
    pub required_features: u64,
    /// Unknown optional bits are preserved for forward compatibility. A reader
    /// must ignore unsupported pruning metadata, never infer absence from it.
    pub optional_features: u64,
}
impl Header {
    pub const fn new(identity: FileIdentity) -> Self {
        Self {
            identity,
            required_features: REQUIRED_FEATURES,
            optional_features: 0,
        }
    }

    pub fn encode(&self) -> Result<[u8; HEADER_SIZE]> {
        self.validate_features()?;
        let mut bytes = [0; HEADER_SIZE];
        bytes[..4].copy_from_slice(b"STV5");
        put_u16(&mut bytes, 4, ENVELOPE_VERSION);
        put_u16(&mut bytes, 6, HEADER_SIZE as u16);
        put_u64(&mut bytes, 8, self.required_features);
        put_u64(&mut bytes, 16, self.optional_features);
        put_u64(&mut bytes, 24, self.identity.table_id.get());
        put_u64(&mut bytes, 32, self.identity.incarnation.get());
        put_u64(&mut bytes, 40, self.identity.volume_id.get());
        put_checksum(&mut bytes);
        Ok(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self> {
        validate_envelope(bytes, b"STV5", HEADER_SIZE)?;
        require_zero(&bytes[48..60])?;
        let header = Self {
            identity: FileIdentity::new(u64_at(bytes, 24), u64_at(bytes, 32), u64_at(bytes, 40))?,
            required_features: u64_at(bytes, 8),
            optional_features: u64_at(bytes, 16),
        };
        header.validate_features()?;
        Ok(header)
    }

    pub const fn known_optional_features(&self) -> u64 {
        self.optional_features & KNOWN_OPTIONAL_FEATURES
    }

    pub(crate) fn validate_features(&self) -> Result<()> {
        let unknown = self.required_features & !KNOWN_REQUIRED_FEATURES;
        if unknown != 0 {
            return Err(EnvelopeError::UnknownRequiredFeatures(unknown));
        }
        let missing = REQUIRED_FEATURES & !self.required_features;
        if missing != 0 {
            return Err(EnvelopeError::MissingRequiredFeatures(missing));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Codec {
    Raw = 0,
    Lz4Block = 1,
}
impl Codec {
    fn decode(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Lz4Block),
            other => Err(EnvelopeError::UnknownCodec(other)),
        }
    }
}

/// Root limits are independent of ordinary-page limits. A reader must pass
/// explicit limits; there is no silently unbounded decode default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadLimits {
    pub root_stored_bytes: u64,
    pub root_decoded_bytes: u64,
    pub page_stored_bytes: u64,
    pub page_decoded_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageDescriptor {
    pub offset: u64,
    pub stored_len: u64,
    pub decoded_len: u64,
    pub stored_checksum: u32,
    pub codec: Codec,
}
impl PageDescriptor {
    /// Encode structural fields; the writer must validate their final file
    /// region once offsets and root position are known.
    pub fn encode(&self) -> Result<[u8; DESCRIPTOR_SIZE]> {
        self.validate_lengths()?;
        let mut bytes = [0; DESCRIPTOR_SIZE];
        put_u64(&mut bytes, 0, self.offset);
        put_u64(&mut bytes, 8, self.stored_len);
        put_u64(&mut bytes, 16, self.decoded_len);
        bytes[24..28].copy_from_slice(&self.stored_checksum.to_le_bytes());
        bytes[28] = self.codec as u8;
        Ok(bytes)
    }

    /// Decode only after the containing directory page's checksum passes.
    /// Before following the pointer, also call Footer::validate_page.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        require_length(bytes, DESCRIPTOR_SIZE)?;
        require_zero(&bytes[29..32])?;
        let page = Self {
            offset: u64_at(bytes, 0),
            stored_len: u64_at(bytes, 8),
            decoded_len: u64_at(bytes, 16),
            stored_checksum: u32_at(bytes, 24),
            codec: Codec::decode(bytes[28])?,
        };
        page.validate_lengths()?;
        Ok(page)
    }

    /// Verify stored bytes before any decompressor or payload parser sees them.
    /// The length must match exactly, including when the slice has a valid
    /// prefix followed by unclaimed bytes. Footer::validate_root/validate_page
    /// and a memory reservation must still precede reading or allocating it.
    pub fn verify_stored_bytes(&self, bytes: &[u8]) -> Result<()> {
        self.validate_lengths()?;
        if u64::try_from(bytes.len()).ok() != Some(self.stored_len) {
            return Err(EnvelopeError::StoredLengthMismatch);
        }
        if crc32fast::hash(bytes) != self.stored_checksum {
            return Err(EnvelopeError::ChecksumMismatch);
        }
        Ok(())
    }

    fn end(&self) -> Result<u64> {
        self.offset
            .checked_add(self.stored_len)
            .ok_or(EnvelopeError::OffsetOverflow)
    }

    fn validate_lengths(&self) -> Result<()> {
        if self.stored_len == 0 || self.decoded_len == 0 {
            return Err(EnvelopeError::EmptyPage);
        }
        if self.codec == Codec::Raw && self.stored_len != self.decoded_len {
            return Err(EnvelopeError::RawLengthMismatch);
        }
        Ok(())
    }

    fn validate_region(&self, end: u64, max_stored: u64, max_decoded: u64) -> Result<()> {
        self.validate_lengths()?;
        if self.stored_len > max_stored {
            return Err(EnvelopeError::StoredLimitExceeded);
        }
        if self.decoded_len > max_decoded {
            return Err(EnvelopeError::DecodedLimitExceeded);
        }
        if self.offset < HEADER_SIZE as u64 || self.end()? > end {
            return Err(EnvelopeError::PageOutOfBounds);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footer {
    pub file_length: u64,
    pub root: PageDescriptor,
}
impl Footer {
    pub fn encode(&self) -> Result<[u8; FOOTER_SIZE]> {
        self.validate_layout()?;
        let mut bytes = [0; FOOTER_SIZE];
        bytes[..4].copy_from_slice(b"V5FT");
        put_u16(&mut bytes, 4, ENVELOPE_VERSION);
        put_u16(&mut bytes, 6, FOOTER_SIZE as u16);
        put_u64(&mut bytes, 8, self.file_length);
        bytes[16..48].copy_from_slice(&self.root.encode()?);
        put_checksum(&mut bytes);
        Ok(bytes)
    }

    /// Validate the envelope against the length of the opened immutable file.
    /// Root bounds/limits are checked before any root read or decompression.
    pub fn decode(bytes: &[u8], file_length: u64, limits: &ReadLimits) -> Result<Self> {
        validate_envelope(bytes, b"V5FT", FOOTER_SIZE)?;
        require_zero(&bytes[48..60])?;
        let footer = Self {
            file_length: u64_at(bytes, 8),
            root: PageDescriptor::decode(&bytes[16..48])?,
        };
        if footer.file_length != file_length {
            return Err(EnvelopeError::FileLengthMismatch);
        }
        footer.validate_root(limits)?;
        Ok(footer)
    }

    pub fn validate_root(&self, limits: &ReadLimits) -> Result<()> {
        self.validate_layout()?;
        self.root.validate_region(
            self.file_length - FOOTER_SIZE as u64,
            limits.root_stored_bytes,
            limits.root_decoded_bytes,
        )
    }

    /// Ordinary data and directory pages cannot overlap the header, root or
    /// footer. Cross-page overlaps and directory cycles require directory state
    /// and are validated by that later reader.
    pub fn validate_page(&self, page: &PageDescriptor, limits: &ReadLimits) -> Result<()> {
        self.validate_root(limits)?;
        page.validate_region(
            self.root.offset,
            limits.page_stored_bytes,
            limits.page_decoded_bytes,
        )
    }

    fn validate_layout(&self) -> Result<()> {
        // Header + footer alone cannot represent even an empty volume.
        if self.file_length <= (HEADER_SIZE + FOOTER_SIZE) as u64 {
            return Err(EnvelopeError::FileTooShort);
        }
        self.root.validate_lengths()?;
        if self.root.offset < HEADER_SIZE as u64 {
            return Err(EnvelopeError::PageOutOfBounds);
        }
        let end = self.root.end()?;
        let footer_start = self.file_length - FOOTER_SIZE as u64;
        if end > footer_start {
            return Err(EnvelopeError::PageOutOfBounds);
        }
        if end != footer_start {
            return Err(EnvelopeError::RootNotLast);
        }
        Ok(())
    }
}

fn require_length(bytes: &[u8], expected: usize) -> Result<()> {
    if bytes.len() != expected {
        Err(EnvelopeError::InvalidLength {
            expected,
            actual: bytes.len(),
        })
    } else {
        Ok(())
    }
}
fn require_zero(bytes: &[u8]) -> Result<()> {
    if bytes.iter().any(|byte| *byte != 0) {
        Err(EnvelopeError::ReservedBytes)
    } else {
        Ok(())
    }
}
fn validate_envelope(bytes: &[u8], magic: &[u8; 4], size: usize) -> Result<()> {
    require_length(bytes, size)?;
    if crc32fast::hash(&bytes[..60]) != u32_at(bytes, 60) {
        return Err(EnvelopeError::ChecksumMismatch);
    }
    if &bytes[..4] != magic {
        return Err(EnvelopeError::InvalidMagic);
    }
    let version = u16_at(bytes, 4);
    if version != ENVELOPE_VERSION {
        return Err(EnvelopeError::UnsupportedVersion(version));
    }
    let declared = u16_at(bytes, 6);
    if usize::from(declared) != size {
        return Err(EnvelopeError::InvalidDeclaredSize(declared));
    }
    Ok(())
}
fn put_checksum(bytes: &mut [u8; 64]) {
    let crc = crc32fast::hash(&bytes[..60]);
    bytes[60..64].copy_from_slice(&crc.to_le_bytes());
}
fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}
fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}
fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(
        bytes[offset..offset + 2]
            .try_into()
            .expect("checked fixed envelope"),
    )
}
fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("checked fixed envelope"),
    )
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("checked fixed envelope"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: ReadLimits = ReadLimits {
        root_stored_bytes: 64,
        root_decoded_bytes: 64,
        page_stored_bytes: 64,
        page_decoded_bytes: 128,
    };

    // Fixed independently with Python struct.pack_into and zlib.crc32; these
    // fixtures are not produced by the encoder under test.
    const GOLDEN_HEADER: &str = "5354563501004000030000000000000001000000000000000807060504030201010000000000000018171615141312110000000000000000000000003f6a2a09";
    const GOLDEN_FOOTER: &str = "5635465401004000e000000000000000800000000000000020000000000000003000000000000000d4c3b2a101000000000000000000000000000000bac95b07";

    fn hex64(text: &str) -> [u8; 64] {
        assert_eq!(text.len(), 128);
        let mut bytes = [0; 64];
        for (target, source) in bytes.iter_mut().zip(text.as_bytes().as_chunks::<2>().0) {
            let digit = |byte| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                _ => panic!("invalid golden digit"),
            };
            *target = digit(source[0]) << 4 | digit(source[1]);
        }
        bytes
    }

    fn golden_footer() -> Footer {
        Footer {
            file_length: 224,
            root: PageDescriptor {
                offset: 128,
                stored_len: 32,
                decoded_len: 48,
                stored_checksum: 0xa1b2_c3d4,
                codec: Codec::Lz4Block,
            },
        }
    }

    #[test]
    fn golden_header_and_footer_bytes_are_stable() {
        let mut header = Header::new(
            FileIdentity::new(0x0102_0304_0506_0708, 1, 0x1112_1314_1516_1718).unwrap(),
        );
        header.optional_features = OPTIONAL_BLOOMS;
        let bytes = hex64(GOLDEN_HEADER);
        assert_eq!(header.encode().unwrap(), bytes);
        assert_eq!(Header::decode(&bytes).unwrap(), header);
        assert_eq!(u32_at(&bytes, 60), 0x092a_6a3f);
        let footer = golden_footer();
        let bytes = hex64(GOLDEN_FOOTER);
        assert_eq!(footer.encode().unwrap(), bytes);
        assert_eq!(Footer::decode(&bytes, 224, &LIMITS).unwrap(), footer);
        assert_eq!(u32_at(&bytes, 60), 0x075b_c9ba);
    }

    #[test]
    fn fixed_parsers_reject_every_truncation_and_trailing_byte() {
        let header = hex64(GOLDEN_HEADER);
        let footer = hex64(GOLDEN_FOOTER);
        for size in 0..64 {
            assert!(matches!(
                Header::decode(&header[..size]),
                Err(EnvelopeError::InvalidLength { .. })
            ));
            assert!(matches!(
                Footer::decode(&footer[..size], 224, &LIMITS),
                Err(EnvelopeError::InvalidLength { .. })
            ));
        }
        let mut extra = [0; 65];
        extra[..64].copy_from_slice(&header);
        assert!(matches!(
            Header::decode(&extra),
            Err(EnvelopeError::InvalidLength { .. })
        ));
        extra[..64].copy_from_slice(&footer);
        assert!(matches!(
            Footer::decode(&extra, 224, &LIMITS),
            Err(EnvelopeError::InvalidLength { .. })
        ));
        let descriptor = golden_footer().root.encode().unwrap();
        for size in 0..32 {
            assert!(PageDescriptor::decode(&descriptor[..size]).is_err());
        }
        let mut extra = [0; 33];
        extra[..32].copy_from_slice(&descriptor);
        assert!(PageDescriptor::decode(&extra).is_err());
    }

    #[test]
    fn every_single_bit_corruption_fails_envelope_checksum() {
        for is_header in [false, true] {
            let original = hex64(if is_header {
                GOLDEN_HEADER
            } else {
                GOLDEN_FOOTER
            });
            for byte in 0..64 {
                for bit in 0..8 {
                    let mut corrupted = original;
                    corrupted[byte] ^= 1 << bit;
                    if is_header {
                        assert_eq!(
                            Header::decode(&corrupted),
                            Err(EnvelopeError::ChecksumMismatch)
                        );
                    } else {
                        assert_eq!(
                            Footer::decode(&corrupted, 224, &LIMITS),
                            Err(EnvelopeError::ChecksumMismatch)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn valid_checksums_do_not_bypass_tag_version_size_or_reserved_checks() {
        for is_header in [false, true] {
            let original = hex64(if is_header {
                GOLDEN_HEADER
            } else {
                GOLDEN_FOOTER
            });
            for (offset, value, error) in [
                (0, b'X', EnvelopeError::InvalidMagic),
                (4, 2, EnvelopeError::UnsupportedVersion(2)),
                (6, 63, EnvelopeError::InvalidDeclaredSize(63)),
                (48, 1, EnvelopeError::ReservedBytes),
                (59, 1, EnvelopeError::ReservedBytes),
            ] {
                let mut bytes = original;
                bytes[offset] = value;
                put_checksum(&mut bytes);
                let actual = if is_header {
                    Header::decode(&bytes).map(|_| ())
                } else {
                    Footer::decode(&bytes, 224, &LIMITS).map(|_| ())
                };
                assert_eq!(actual, Err(error));
            }
        }
    }

    #[test]
    fn mandatory_features_and_identity_are_checked_but_optional_bits_survive() {
        let original = hex64(GOLDEN_HEADER);
        for required in [
            0,
            REQUIRED_SOURCE_LSNS,
            REQUIRED_GROUP_RANGES,
            REQUIRED_FEATURES | (1 << 63),
        ] {
            let mut bytes = original;
            put_u64(&mut bytes, 8, required);
            put_checksum(&mut bytes);
            let error = if required & !KNOWN_REQUIRED_FEATURES != 0 {
                EnvelopeError::UnknownRequiredFeatures(1 << 63)
            } else {
                EnvelopeError::MissingRequiredFeatures(REQUIRED_FEATURES & !required)
            };
            assert_eq!(Header::decode(&bytes), Err(error));
            let mut header = Header::decode(&original).unwrap();
            header.required_features = required;
            assert_eq!(header.encode(), Err(error));
        }
        for (offset, field) in [
            (24, IdentityField::Table),
            (32, IdentityField::Incarnation),
            (40, IdentityField::Volume),
        ] {
            let mut bytes = original;
            put_u64(&mut bytes, offset, 0);
            put_checksum(&mut bytes);
            assert_eq!(
                Header::decode(&bytes),
                Err(EnvelopeError::ZeroIdentity(field))
            );
        }
        let mut bytes = original;
        put_u64(&mut bytes, 16, OPTIONAL_BLOOMS | (1 << 63));
        put_checksum(&mut bytes);
        let header = Header::decode(&bytes).unwrap();
        assert_eq!(header.optional_features, OPTIONAL_BLOOMS | (1 << 63));
        assert_eq!(header.known_optional_features(), OPTIONAL_BLOOMS);
        assert_eq!(header.encode().unwrap(), bytes);
    }

    #[test]
    fn explicit_codec_does_not_depend_on_equal_lengths() {
        for codec in [Codec::Raw, Codec::Lz4Block] {
            let page = PageDescriptor {
                offset: 64,
                stored_len: 9,
                decoded_len: 9,
                stored_checksum: 0xcbf4_3926,
                codec,
            };
            let bytes = page.encode().unwrap();
            assert_eq!(PageDescriptor::decode(&bytes).unwrap(), page);
            assert_eq!(bytes[28], codec as u8);
            page.verify_stored_bytes(b"123456789").unwrap();
            assert_eq!(
                page.verify_stored_bytes(b"123456788"),
                Err(EnvelopeError::ChecksumMismatch)
            );
            assert!(page.verify_stored_bytes(b"12345678").is_err());
            assert!(page.verify_stored_bytes(b"1234567890").is_err());
        }
        let empty = PageDescriptor {
            offset: 64,
            stored_len: 0,
            decoded_len: 0,
            stored_checksum: 0,
            codec: Codec::Raw,
        };
        assert_eq!(
            empty.verify_stored_bytes(&[]),
            Err(EnvelopeError::EmptyPage)
        );
        let invalid_raw = PageDescriptor {
            stored_len: 9,
            decoded_len: 10,
            ..empty
        };
        assert_eq!(
            invalid_raw.verify_stored_bytes(b"123456789"),
            Err(EnvelopeError::RawLengthMismatch)
        );
        let mut descriptor = golden_footer().root.encode().unwrap();
        descriptor[28] = 2;
        assert_eq!(
            PageDescriptor::decode(&descriptor),
            Err(EnvelopeError::UnknownCodec(2))
        );
        descriptor[28] = Codec::Raw as u8;
        assert_eq!(
            PageDescriptor::decode(&descriptor),
            Err(EnvelopeError::RawLengthMismatch)
        );
        descriptor[28] = Codec::Lz4Block as u8;
        for position in 29..32 {
            let mut invalid = descriptor;
            invalid[position] = 1;
            assert_eq!(
                PageDescriptor::decode(&invalid),
                Err(EnvelopeError::ReservedBytes)
            );
        }
    }

    #[test]
    fn root_and_page_budgets_are_independent_and_inclusive() {
        let footer = golden_footer();
        let page = PageDescriptor {
            offset: 64,
            stored_len: 64,
            decoded_len: 128,
            codec: Codec::Lz4Block,
            stored_checksum: 0,
        };
        footer.validate_page(&page, &LIMITS).unwrap();
        let mut limits = LIMITS;
        limits.root_stored_bytes = 31;
        assert_eq!(
            footer.validate_root(&limits),
            Err(EnvelopeError::StoredLimitExceeded)
        );
        limits.root_stored_bytes = 32;
        limits.root_decoded_bytes = 47;
        assert_eq!(
            footer.validate_root(&limits),
            Err(EnvelopeError::DecodedLimitExceeded)
        );
        limits.root_decoded_bytes = 48;
        footer.validate_root(&limits).unwrap();
        limits.page_stored_bytes = 63;
        assert_eq!(
            footer.validate_page(&page, &limits),
            Err(EnvelopeError::StoredLimitExceeded)
        );
        limits.page_stored_bytes = 64;
        limits.page_decoded_bytes = 127;
        assert_eq!(
            footer.validate_page(&page, &limits),
            Err(EnvelopeError::DecodedLimitExceeded)
        );
        // Limiting ordinary pages does not prevent loading a valid bounded root.
        footer.validate_root(&limits).unwrap();
        let bytes = footer.encode().unwrap();
        assert_eq!(
            Footer::decode(&bytes, 225, &LIMITS),
            Err(EnvelopeError::FileLengthMismatch)
        );
    }

    #[test]
    fn root_is_nonempty_last_and_cannot_overlap_header_or_footer() {
        let valid = golden_footer();
        for size in 0..=128 {
            let invalid = Footer {
                file_length: size,
                ..valid
            };
            assert_eq!(invalid.encode(), Err(EnvelopeError::FileTooShort));
            let mut bytes = valid.encode().unwrap();
            put_u64(&mut bytes, 8, size);
            put_checksum(&mut bytes);
            assert_eq!(
                Footer::decode(&bytes, size, &LIMITS),
                Err(EnvelopeError::FileTooShort)
            );
        }
        for zero_stored in [false, true] {
            let mut invalid = valid;
            if zero_stored {
                invalid.root.stored_len = 0;
            } else {
                invalid.root.decoded_len = 0;
            }
            assert_eq!(invalid.encode(), Err(EnvelopeError::EmptyPage));
        }
        for (offset, error) in [
            (0, EnvelopeError::PageOutOfBounds),
            (63, EnvelopeError::PageOutOfBounds),
            (127, EnvelopeError::RootNotLast),
            (129, EnvelopeError::PageOutOfBounds),
            (u64::MAX - 1, EnvelopeError::OffsetOverflow),
        ] {
            let mut invalid = valid;
            invalid.root.offset = offset;
            assert_eq!(invalid.encode(), Err(error));
        }
        let smallest = Footer {
            file_length: 129,
            root: PageDescriptor {
                offset: 64,
                stored_len: 1,
                decoded_len: 1,
                stored_checksum: 0,
                codec: Codec::Raw,
            },
        };
        Footer::decode(&smallest.encode().unwrap(), 129, &LIMITS).unwrap();
    }

    #[test]
    fn pages_cannot_alias_header_root_or_footer_and_address_math_is_checked() {
        let footer = golden_footer();
        let valid = PageDescriptor {
            offset: 64,
            stored_len: 9,
            decoded_len: 9,
            stored_checksum: 0,
            codec: Codec::Raw,
        };
        footer.validate_page(&valid, &LIMITS).unwrap();
        for offset in [0, 63, 120, 128, 159, 160, 224] {
            let invalid = PageDescriptor { offset, ..valid };
            assert_eq!(
                footer.validate_page(&invalid, &LIMITS),
                Err(EnvelopeError::PageOutOfBounds)
            );
        }
        let overflow = PageDescriptor {
            offset: u64::MAX - 1,
            ..valid
        };
        assert_eq!(
            footer.validate_page(&overflow, &LIMITS),
            Err(EnvelopeError::OffsetOverflow)
        );
        let mut invalid = valid;
        invalid.stored_len = u64::MAX;
        invalid.decoded_len = u64::MAX;
        assert_eq!(
            footer.validate_page(&invalid, &LIMITS),
            Err(EnvelopeError::StoredLimitExceeded)
        );
        let mut maximal = footer;
        maximal.file_length = u64::MAX;
        maximal.root.offset = u64::MAX - 64 - maximal.root.stored_len;
        Footer::decode(&maximal.encode().unwrap(), u64::MAX, &LIMITS).unwrap();
    }
}
