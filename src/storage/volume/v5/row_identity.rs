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

//! Separate row-ID (V5RI) and source-LSN (V5LS) pages, in physical row order.
//! Revision 1 header32: magic[0..4], revision:u16[4..6], size:u16[6..8],
//! group:u64[8..16], row_start:u64[16..24], row_count:u32[24..28], layout:u8[28],
//! lane_width:u8[29]=8, reserved_zero[30..32]. The body is exactly rows*8 LE
//! lanes. Directory keys use (RowIds/SourceLsns, GLOBAL_COLUMN, group).
//! IdentityPageExpectation comes from checked group metadata and provides the
//! exact decoded byte requirement before buffer reservation and page I/O.
//! CRC and exact decompression precede these borrowed decoders. No operation
//! allocates or owns a buffer; no typed reference casts assume aligned input.
//!
//! Row IDs have exact advertised extrema; RowId layout is strictly ascending.
//! Clustered pages retain physical order and make no global uniqueness claim.
//! Group sequence coverage and locator bijection remain separate validators.
//!
//! Nonzero source lanes are actual DML LSNs. Zero is never absent, memory-only,
//! or unlogged data: it names LegacyBase(E,G), and decoding requires an explicit
//! caller assertion of matching durable installer evidence. Decoding metadata
//! or calling its constructor cannot certify that evidence. The assertion is a
//! trust boundary, not a cryptographic proof or a disk verification routine.
//!
//! Phase5 canonicalizes the COMPLETE recovered canonical state staged into E
//! as LegacyBase(E,G), resolving old-cold/recovered-hot/deletion precedence
//! before publishing E. Numeric sources in a context volume must be >G. This
//! avoids misordering recovered pre-barrier updates against unchanged legacy
//! rows, without fabricating a numeric row LSN. Conversion retains E/G; sources
//! from different base contexts cannot be combined into one volume. Context may
//! remain conservatively present when a selected page has only new DML sources.
//! A new memory-only row has no representable source until the lifecycle layer
//! supplies a real durable source; it cannot silently become the zero marker.

use std::fmt;
use std::num::NonZeroU64;

use super::directory::{DirectoryError, Layout, RootSummary, RowBounds};
use super::envelope::{
    FileIdentity, Footer, Header, LegacyBase, PageDescriptor, ReadLimits, REQUIRED_LEGACY_BASE,
};
use super::group_metadata::{layout_tag, GroupExpectation};
use super::page_io::{PageIoError, PageReadPlan};

pub const IDENTITY_HEADER_BYTES: usize = 32;
pub const IDENTITY_LANE_BYTES: usize = 8;
pub const MAX_IDENTITY_PAGE_BYTES: usize = IDENTITY_HEADER_BYTES + 4096 * IDENTITY_LANE_BYTES;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityError {
    Directory(DirectoryError),
    Length,
    Magic,
    Revision,
    Reserved,
    Identity,
    Layout,
    LaneWidth,
    RowOrder,
    RowBounds,
    OutputTooShort,
    MissingEvidence,
    UnexpectedEvidence,
    EvidenceMismatch,
    MissingLegacyBase,
    LsnBeforeBarrier,
    BaseContextMismatch,
}
impl fmt::Display for IdentityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid V5 row identity: {self:?}")
    }
}
impl std::error::Error for IdentityError {}
impl From<DirectoryError> for IdentityError {
    fn from(error: DirectoryError) -> Self {
        Self::Directory(error)
    }
}
type Result<T> = std::result::Result<T, IdentityError>;

/// The caller's explicit assertion that its durable catalog/upgrade-root check
/// has already verified this exact file identity and checkpoint. This type does
/// not inspect disk or establish durability itself; do not construct it merely
/// from the header/root you are trying to validate.
#[derive(Clone, Copy, Debug)]
pub struct VerifiedLegacyBase {
    identity: FileIdentity,
    base: LegacyBase,
}
impl VerifiedLegacyBase {
    /// Trusted installer boundary. `identity` must be the expected manifest file
    /// identity; `base` must come from verified complete upgrade-root evidence,
    /// not unchecked volume metadata or the current WAL LSN. The caller asserts
    /// those checks have completed. No production caller exists in this stage.
    pub const fn assert_verified_checkpoint(identity: FileIdentity, base: LegacyBase) -> Self {
        Self { identity, base }
    }
}

/// Encoding-only assertion: the lifecycle coordinator assigned this file to
/// the complete canonical checkpoint captured through G under generation E.
/// The checkpoint may still be staged. This is not installed/durable evidence
/// and cannot authorize a source-page decoder.
#[derive(Clone, Copy, Debug)]
pub struct PlannedCheckpoint {
    identity: FileIdentity,
    base: LegacyBase,
}
impl PlannedCheckpoint {
    /// The caller must have captured the complete quiescent checkpoint and its
    /// exact E/G identity. Encoding alone does not complete or publish it.
    pub const fn assert_captured_checkpoint(identity: FileIdentity, base: LegacyBase) -> Self {
        Self { identity, base }
    }
}

/// Source policy for emission, separate from installed-file read authorization.
/// No directory/root pointer is required before the payloads are written.
#[derive(Clone, Copy, Debug)]
pub struct SourceEncodingContext {
    base: Option<LegacyBase>,
}
impl SourceEncodingContext {
    pub fn for_staged_file(
        header: &Header,
        checkpoint: Option<&PlannedCheckpoint>,
    ) -> Result<Self> {
        header.validate_features().map_err(DirectoryError::from)?;
        match (
            header.required_features & REQUIRED_LEGACY_BASE != 0,
            checkpoint,
        ) {
            (false, None) => Ok(Self { base: None }),
            (false, Some(_)) => Err(IdentityError::UnexpectedEvidence),
            (true, None) => Err(IdentityError::MissingEvidence),
            (true, Some(checkpoint)) if checkpoint.identity == header.identity => Ok(Self {
                base: Some(checkpoint.base),
            }),
            _ => Err(IdentityError::EvidenceMismatch),
        }
    }
    pub const fn legacy_base(self) -> Option<LegacyBase> {
        self.base
    }
    #[inline]
    fn validate_source(self, source: RowSource) -> Result<u64> {
        match source {
            RowSource::LegacyBase => self.base.map(|_| 0).ok_or(IdentityError::MissingLegacyBase),
            RowSource::Dml(lsn) if self.base.is_some_and(|base| lsn.get() <= base.barrier_lsn) => {
                Err(IdentityError::LsnBeforeBarrier)
            }
            RowSource::Dml(lsn) => Ok(lsn.get()),
        }
    }
}
impl From<SourceContext> for SourceEncodingContext {
    fn from(context: SourceContext) -> Self {
        Self { base: context.base }
    }
}

/// Bound source interpretation for one already identity-checked volume. The
/// supplied header/root and each following page must be from that same file.
#[derive(Clone, Copy, Debug)]
pub struct SourceContext {
    base: Option<LegacyBase>,
}
impl SourceContext {
    pub fn bind(
        header: &Header,
        root: &RootSummary,
        evidence: Option<&VerifiedLegacyBase>,
    ) -> Result<Self> {
        root.validate_header(header)?;
        match (root.legacy_base, evidence) {
            (None, None) => Ok(Self { base: None }),
            (None, Some(_)) => Err(IdentityError::UnexpectedEvidence),
            (Some(_), None) => Err(IdentityError::MissingEvidence),
            (Some(base), Some(proof))
                if base == proof.base && header.identity == proof.identity =>
            {
                Ok(Self { base: Some(base) })
            }
            _ => Err(IdentityError::EvidenceMismatch),
        }
    }
    pub const fn legacy_base(self) -> Option<LegacyBase> {
        self.base
    }
    fn validate_source(self, source: RowSource) -> Result<u64> {
        SourceEncodingContext::from(self).validate_source(source)
    }
}

/// Deliberately no From<u64>, Default, or unlogged variant: zero requires an
/// explicit LegacyBase choice and a separately bound source context.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSource {
    Dml(NonZeroU64),
    /// Explicit bootstrap/retained-provenance assertion relative to the bound
    /// read or encoding context. Never derive this marker from a missing new-row LSN.
    LegacyBase,
}

#[derive(Clone, Copy, Debug)]
pub struct IdentityPageExpectation {
    group: GroupExpectation,
}
impl IdentityPageExpectation {
    pub const fn new(group: GroupExpectation) -> Self {
        Self { group }
    }
    pub const fn decoded_len(self) -> usize {
        IDENTITY_HEADER_BYTES + self.group.record().row_count as usize * IDENTITY_LANE_BYTES
    }
    pub fn read_plan(
        self,
        footer: &Footer,
        descriptor: PageDescriptor,
        limits: &ReadLimits,
    ) -> std::result::Result<PageReadPlan, PageIoError> {
        if descriptor.decoded_len != self.decoded_len() as u64 {
            return Err(PageIoError::PayloadLength);
        }
        PageReadPlan::for_page(footer, descriptor, limits)
    }
    pub fn decode_row_ids(self, bytes: &[u8]) -> Result<RowIdPageRef<'_>> {
        self.validate_header(bytes, b"V5RI")?;
        let view = RowIdPageRef {
            bytes: &bytes[IDENTITY_HEADER_BYTES..],
        };
        self.validate_rows(view.iter())?;
        Ok(view)
    }
    pub fn encode_row_ids(self, row_ids: &[i64], output: &mut [u8]) -> Result<usize> {
        self.preflight(row_ids.len(), output.len())?;
        self.validate_rows(row_ids.iter().copied())?;
        self.encode_header(output, b"V5RI");
        for (id, lane) in row_ids.iter().zip(
            output[IDENTITY_HEADER_BYTES..self.decoded_len()]
                .as_chunks_mut::<8>()
                .0,
        ) {
            *lane = id.to_le_bytes();
        }
        Ok(self.decoded_len())
    }
    pub fn decode_sources(
        self,
        bytes: &[u8],
        context: SourceContext,
    ) -> Result<SourceLsnPageRef<'_>> {
        self.validate_header(bytes, b"V5LS")?;
        let view = SourceLsnPageRef {
            bytes: &bytes[IDENTITY_HEADER_BYTES..],
            context,
        };
        for source in view.iter() {
            context.validate_source(source)?;
        }
        Ok(view)
    }
    /// Encode caller-owned sources. LegacyBase markers explicitly assert the
    /// bound checkpoint provenance (the bootstrap path); decoded page conversion
    /// should prefer encode_sources_from_page, which checks that provenance.
    pub fn encode_sources(
        self,
        sources: &[RowSource],
        context: impl Into<SourceEncodingContext>,
        output: &mut [u8],
    ) -> Result<usize> {
        self.preflight(sources.len(), output.len())?;
        let context = context.into();
        for source in sources {
            context.validate_source(*source)?;
        }
        self.encode_header(output, b"V5LS");
        for (source, lane) in sources.iter().zip(
            output[IDENTITY_HEADER_BYTES..self.decoded_len()]
                .as_chunks_mut::<8>()
                .0,
        ) {
            *lane = match source {
                RowSource::Dml(lsn) => lsn.get(),
                RowSource::LegacyBase => 0,
            }
            .to_le_bytes();
        }
        Ok(self.decoded_len())
    }
    /// Conversion of a previously decoded page preserves its provenance without
    /// a per-row E/G sidecar. Reject a different existing source base even if
    /// this page's selected lanes happen to be numeric. A base-free source may
    /// join a context volume when every numeric LSN is above its barrier. A
    /// merger must retain the analogous
    /// source context for every input; extracting only zero marker lanes and
    /// relabeling them with a different base is not a valid conversion.
    pub fn encode_sources_from_page(
        self,
        source: SourceLsnPageRef<'_>,
        context: impl Into<SourceEncodingContext>,
        output: &mut [u8],
    ) -> Result<usize> {
        self.preflight(source.bytes.len() / IDENTITY_LANE_BYTES, output.len())?;
        let context = context.into();
        if source.context.base.is_some() && source.context.base != context.base {
            return Err(IdentityError::BaseContextMismatch);
        }
        for value in source.iter() {
            context.validate_source(value)?;
        }
        self.encode_header(output, b"V5LS");
        output[IDENTITY_HEADER_BYTES..self.decoded_len()].copy_from_slice(source.bytes);
        Ok(self.decoded_len())
    }
    fn preflight(self, count: usize, output_len: usize) -> Result<()> {
        if count != self.group.record().row_count as usize {
            return Err(IdentityError::Length);
        }
        if output_len < self.decoded_len() {
            return Err(IdentityError::OutputTooShort);
        }
        Ok(())
    }
    fn validate_rows(self, rows: impl Iterator<Item = i64>) -> Result<()> {
        let mut previous = None;
        let mut bounds: Option<RowBounds> = None;
        for row in rows {
            if self.group.layout() == Layout::RowId && previous.is_some_and(|old| old >= row) {
                return Err(IdentityError::RowOrder);
            }
            bounds = Some(match bounds {
                None => RowBounds { min: row, max: row },
                Some(old) => RowBounds {
                    min: old.min.min(row),
                    max: old.max.max(row),
                },
            });
            previous = Some(row);
        }
        if bounds != Some(self.group.record().rows) {
            return Err(IdentityError::RowBounds);
        }
        Ok(())
    }
    fn validate_header(self, bytes: &[u8], magic: &[u8; 4]) -> Result<()> {
        if bytes.len() != self.decoded_len() {
            return Err(IdentityError::Length);
        }
        if &bytes[..4] != magic {
            return Err(IdentityError::Magic);
        }
        if u16_at(bytes, 4) != 1 {
            return Err(IdentityError::Revision);
        }
        if u16_at(bytes, 6) != IDENTITY_HEADER_BYTES as u16 {
            return Err(IdentityError::Length);
        }
        if u64_at(bytes, 8) != self.group.group()
            || u64_at(bytes, 16) != self.group.record().row_start
            || u32_at(bytes, 24) != self.group.record().row_count
        {
            return Err(IdentityError::Identity);
        }
        if bytes[28] != layout_tag(self.group.layout()) {
            return Err(IdentityError::Layout);
        }
        if bytes[29] != IDENTITY_LANE_BYTES as u8 {
            return Err(IdentityError::LaneWidth);
        }
        if bytes[30..32] != [0, 0] {
            return Err(IdentityError::Reserved);
        }
        Ok(())
    }
    fn encode_header(self, bytes: &mut [u8], magic: &[u8; 4]) {
        bytes[..IDENTITY_HEADER_BYTES].fill(0);
        bytes[..4].copy_from_slice(magic);
        bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
        bytes[6..8].copy_from_slice(&(IDENTITY_HEADER_BYTES as u16).to_le_bytes());
        bytes[8..16].copy_from_slice(&self.group.group().to_le_bytes());
        bytes[16..24].copy_from_slice(&self.group.record().row_start.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.group.record().row_count.to_le_bytes());
        bytes[28] = layout_tag(self.group.layout());
        bytes[29] = IDENTITY_LANE_BYTES as u8;
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RowIdPageRef<'a> {
    bytes: &'a [u8],
}
impl RowIdPageRef<'_> {
    pub fn iter(&self) -> impl ExactSizeIterator<Item = i64> + '_ {
        self.bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
    }
    pub fn get(&self, row: usize) -> Option<i64> {
        self.bytes
            .as_chunks::<8>()
            .0
            .get(row)
            .map(|b| i64::from_le_bytes(*b))
    }
}
#[derive(Clone, Copy, Debug)]
pub struct SourceLsnPageRef<'a> {
    bytes: &'a [u8],
    context: SourceContext,
}
impl SourceLsnPageRef<'_> {
    pub fn iter(&self) -> impl ExactSizeIterator<Item = RowSource> + '_ {
        self.bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| source(u64::from_le_bytes(*b)))
    }
    pub fn get(&self, row: usize) -> Option<RowSource> {
        self.bytes
            .as_chunks::<8>()
            .0
            .get(row)
            .map(|b| source(u64::from_le_bytes(*b)))
    }
    pub const fn legacy_base(&self) -> Option<LegacyBase> {
        self.context.base
    }
}
fn source(raw: u64) -> RowSource {
    NonZeroU64::new(raw).map_or(RowSource::LegacyBase, RowSource::Dml)
}
fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}
fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

#[cfg(test)]
mod tests {
    use super::super::envelope::{Codec, REQUIRED_LEGACY_BASE};
    use super::super::group_metadata::{
        tests::{fixed_hex, root},
        GroupRecord,
    };
    use super::*;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }
    fn expected(layout: Layout) -> (RootSummary, IdentityPageExpectation) {
        let bounds = RowBounds {
            min: i64::MIN,
            max: i64::MAX,
        };
        let root = root(layout, 3, 1, bounds);
        let group = GroupExpectation::new(
            &root,
            0,
            GroupRecord {
                row_start: 0,
                row_count: 3,
                column_count: 2,
                rows: bounds,
            },
        )
        .unwrap();
        (root, IdentityPageExpectation::new(group))
    }
    fn base_context(
        mut root: RootSummary,
        barrier_lsn: u64,
    ) -> (Header, RootSummary, VerifiedLegacyBase, SourceContext) {
        let mut header = Header::new(FileIdentity::new(11, 1, 22).unwrap());
        header.required_features |= REQUIRED_LEGACY_BASE;
        let base = LegacyBase {
            generation: nz(7),
            barrier_lsn,
        };
        root.legacy_base = Some(base);
        let evidence = VerifiedLegacyBase::assert_verified_checkpoint(header.identity, base);
        let context = SourceContext::bind(&header, &root, Some(&evidence)).unwrap();
        (header, root, evidence, context)
    }
    #[test]
    fn staged_checkpoint_encoding_does_not_supply_installed_read_evidence() {
        let (root, expected) = expected(Layout::RowId);
        let mut header = Header::new(FileIdentity::new(11, 1, 22).unwrap());
        let plain = SourceEncodingContext::for_staged_file(&header, None).unwrap();
        let base = LegacyBase {
            generation: nz(7),
            barrier_lsn: 7,
        };
        let checkpoint = PlannedCheckpoint::assert_captured_checkpoint(header.identity, base);
        assert!(matches!(
            SourceEncodingContext::for_staged_file(&header, Some(&checkpoint)),
            Err(IdentityError::UnexpectedEvidence)
        ));
        header.required_features |= REQUIRED_LEGACY_BASE;
        assert!(matches!(
            SourceEncodingContext::for_staged_file(&header, None),
            Err(IdentityError::MissingEvidence)
        ));
        let staged = SourceEncodingContext::for_staged_file(&header, Some(&checkpoint)).unwrap();
        assert_eq!(staged.legacy_base(), Some(base));
        let sources = [
            RowSource::LegacyBase,
            RowSource::Dml(nz(8)),
            RowSource::Dml(nz(u64::MAX)),
        ];
        let mut output = [0x5a; 56];
        expected
            .encode_sources(&sources, staged, &mut output)
            .unwrap();
        let with_base = RootSummary {
            legacy_base: Some(base),
            ..root
        };
        assert!(matches!(
            SourceContext::bind(&header, &with_base, None),
            Err(IdentityError::MissingEvidence)
        ));
        let evidence = VerifiedLegacyBase::assert_verified_checkpoint(header.identity, base);
        let installed = SourceContext::bind(&header, &with_base, Some(&evidence)).unwrap();
        assert!(expected
            .decode_sources(&output, installed)
            .unwrap()
            .iter()
            .eq(sources));
        for context in [plain, staged] {
            output.fill(0x5a);
            let invalid = if context.legacy_base().is_some() {
                RowSource::Dml(nz(7))
            } else {
                RowSource::LegacyBase
            };
            assert!(expected
                .encode_sources(&[invalid; 3], context, &mut output)
                .is_err());
            assert_eq!(output, [0x5a; 56]);
        }
        for identity in [
            FileIdentity::new(12, 1, 22).unwrap(),
            FileIdentity::new(11, 2, 22).unwrap(),
            FileIdentity::new(11, 1, 23).unwrap(),
        ] {
            let wrong = PlannedCheckpoint::assert_captured_checkpoint(identity, base);
            assert!(matches!(
                SourceEncodingContext::for_staged_file(&header, Some(&wrong)),
                Err(IdentityError::EvidenceMismatch)
            ));
        }
        header.required_features |= 1 << 63;
        assert!(SourceEncodingContext::for_staged_file(&header, Some(&checkpoint)).is_err());
    }
    #[test]
    fn identity_independent_wire_goldens_and_signed_zero_extrema() {
        // Python struct/zlib vectors independent of these encoders.
        let row_golden = fixed_hex::<56>("563552490100200000000000000000000000000000000000030000000008000000000000000000800000000000000000ffffffffffffff7f");
        let lsn_golden = fixed_hex::<56>("56354c530100200000000000000000000000000000000000030000000008000000000000000000000800000000000000ffffffffffffffff");
        assert_eq!(crc32fast::hash(&row_golden), 0xe792_c2b0);
        assert_eq!(crc32fast::hash(&lsn_golden), 0xecc6_c54f);
        let (root, expected) = expected(Layout::RowId);
        let (_, _, _, context) = base_context(root, 7);
        let mut output = [0x5a; 64];
        let ids = [i64::MIN, 0, i64::MAX];
        assert_eq!(expected.encode_row_ids(&ids, &mut output).unwrap(), 56);
        assert_eq!(output[..56], row_golden);
        assert_eq!(output[56..], [0x5a; 8]);
        let view = expected.decode_row_ids(&row_golden).unwrap();
        assert!(view.iter().eq(ids));
        assert_eq!(view.get(1), Some(0));
        assert_eq!(view.get(3), None);
        let sources = [
            RowSource::LegacyBase,
            RowSource::Dml(nz(8)),
            RowSource::Dml(nz(u64::MAX)),
        ];
        assert_eq!(std::mem::size_of::<RowSource>(), 8);
        expected
            .encode_sources(&sources, context, &mut output)
            .unwrap();
        assert_eq!(output[..56], lsn_golden);
        assert_eq!(output[56..], [0x5a; 8]);
        let view = expected.decode_sources(&lsn_golden, context).unwrap();
        assert!(view.iter().eq(sources));
        assert_eq!(view.get(0), Some(RowSource::LegacyBase));
        assert_eq!(view.get(3), None);
        assert_eq!(view.legacy_base(), context.legacy_base());
        for len in 0..56 {
            assert!(expected.decode_row_ids(&row_golden[..len]).is_err());
            assert!(expected
                .decode_sources(&lsn_golden[..len], context)
                .is_err());
        }
        assert!(expected.decode_row_ids(&output).is_err());
    }
    #[test]
    fn identity_context_requires_matching_installer_assertion_not_metadata() {
        let (original, _) = expected(Layout::RowId);
        let (header, root, evidence, _) = base_context(original, 0);
        assert!(matches!(
            SourceContext::bind(&header, &root, None),
            Err(IdentityError::MissingEvidence)
        ));
        let plain_header = Header::new(header.identity);
        assert!(SourceContext::bind(&plain_header, &original, None).is_ok());
        assert!(matches!(
            SourceContext::bind(&plain_header, &original, Some(&evidence)),
            Err(IdentityError::UnexpectedEvidence)
        ));
        assert!(matches!(
            SourceContext::bind(&plain_header, &root, Some(&evidence)),
            Err(IdentityError::Directory(DirectoryError::LegacyBase))
        ));
        assert!(matches!(
            SourceContext::bind(&header, &original, Some(&evidence)),
            Err(IdentityError::Directory(DirectoryError::LegacyBase))
        ));
        for identity in [
            FileIdentity::new(12, 1, 22).unwrap(),
            FileIdentity::new(11, 2, 22).unwrap(),
            FileIdentity::new(11, 1, 23).unwrap(),
        ] {
            let mismatch =
                VerifiedLegacyBase::assert_verified_checkpoint(identity, root.legacy_base.unwrap());
            assert!(matches!(
                SourceContext::bind(&header, &root, Some(&mismatch)),
                Err(IdentityError::EvidenceMismatch)
            ));
        }
        for base in [
            LegacyBase {
                generation: nz(8),
                barrier_lsn: 0,
            },
            LegacyBase {
                generation: nz(7),
                barrier_lsn: 1,
            },
        ] {
            let mismatch = VerifiedLegacyBase::assert_verified_checkpoint(header.identity, base);
            assert!(matches!(
                SourceContext::bind(&header, &root, Some(&mismatch)),
                Err(IdentityError::EvidenceMismatch)
            ));
        }
    }
    #[test]
    fn identity_conversion_preserves_exact_legacy_context_and_output_on_mismatch() {
        let (root, expected) = expected(Layout::RowId);
        let (_, _, _, context) = base_context(root, 7);
        let mut source_bytes = [0; 56];
        expected
            .encode_sources(
                &[
                    RowSource::LegacyBase,
                    RowSource::Dml(nz(8)),
                    RowSource::Dml(nz(u64::MAX)),
                ],
                context,
                &mut source_bytes,
            )
            .unwrap();
        let source = expected.decode_sources(&source_bytes, context).unwrap();
        let mut output = [0x5a; 64];
        expected
            .encode_sources_from_page(source, context, &mut output)
            .unwrap();
        assert_eq!(output[..56], source_bytes);
        assert_eq!(output[56..], [0x5a; 8]);
        let (header, mut other_root, _, _) = base_context(root, 7);
        for base in [
            LegacyBase {
                generation: nz(8),
                barrier_lsn: 7,
            },
            LegacyBase {
                generation: nz(7),
                barrier_lsn: 6,
            },
        ] {
            other_root.legacy_base = Some(base);
            let proof = VerifiedLegacyBase::assert_verified_checkpoint(header.identity, base);
            let other = SourceContext::bind(&header, &other_root, Some(&proof)).unwrap();
            output.fill(0x5a);
            assert_eq!(
                expected.encode_sources_from_page(source, other, &mut output),
                Err(IdentityError::BaseContextMismatch)
            );
            assert_eq!(output, [0x5a; 64]);
        }
        let plain = SourceContext::bind(&Header::new(header.identity), &root, None).unwrap();
        assert_eq!(
            expected.encode_sources_from_page(source, plain, &mut output),
            Err(IdentityError::BaseContextMismatch)
        );
        // Post-upgrade all-DML volumes can join legacy-backed compaction. No
        // zero is fabricated and every numeric source must exceed the barrier.
        for value in [7, 8, u64::MAX] {
            expected
                .encode_sources(&[RowSource::Dml(nz(value)); 3], plain, &mut source_bytes)
                .unwrap();
            let fresh = expected.decode_sources(&source_bytes, plain).unwrap();
            output.fill(0x5a);
            let result = expected.encode_sources_from_page(fresh, context, &mut output);
            if value == 7 {
                assert_eq!(result, Err(IdentityError::LsnBeforeBarrier));
                assert_eq!(output, [0x5a; 64]);
            } else {
                result.unwrap();
                assert_eq!(output[..56], source_bytes);
                assert_eq!(output[56..], [0x5a; 8]);
                assert!(expected
                    .decode_sources(&output[..56], context)
                    .unwrap()
                    .iter()
                    .all(|source| source == RowSource::Dml(nz(value))));
            }
        }
    }

    #[test]
    fn identity_sources_enforce_barrier_at_both_encode_and_decode() {
        let (root, expected) = expected(Layout::RowId);
        let plain = SourceContext::bind(
            &Header::new(FileIdentity::new(1, 1, 1).unwrap()),
            &root,
            None,
        )
        .unwrap();
        let mut output = [0x5a; 56];
        assert_eq!(
            expected.encode_sources(&[RowSource::LegacyBase; 3], plain, &mut output),
            Err(IdentityError::MissingLegacyBase)
        );
        assert_eq!(output, [0x5a; 56]);
        for barrier in [0, 1, 7, u64::MAX] {
            let (_, _, _, context) = base_context(root, barrier);
            for value in [0, 1, 7, 8, u64::MAX] {
                let source = source(value);
                let valid = value == 0 || value > barrier;
                let mut bytes = [0; 56];
                let result = expected.encode_sources(&[source; 3], context, &mut bytes);
                assert_eq!(result.is_ok(), valid, "barrier={barrier} value={value}");
                expected
                    .encode_sources(&[RowSource::LegacyBase; 3], context, &mut bytes)
                    .unwrap();
                for lane in bytes[32..].as_chunks_mut::<8>().0 {
                    *lane = value.to_le_bytes();
                }
                assert_eq!(expected.decode_sources(&bytes, context).is_ok(), valid);
                if value == 0 {
                    assert!(matches!(
                        expected.decode_sources(&bytes, plain),
                        Err(IdentityError::MissingLegacyBase)
                    ));
                }
            }
        }
        expected
            .encode_sources(&[RowSource::Dml(nz(1)); 3], plain, &mut output)
            .unwrap();
        assert!(expected.decode_sources(&output, plain).is_ok());
    }
    #[test]
    fn identity_row_order_layout_and_header_corruption() {
        let (_, rowid) = expected(Layout::RowId);
        let (_, clustered) = expected(Layout::Clustered);
        let unordered = [i64::MAX, 0, i64::MIN];
        let mut output = [0x5a; 56];
        assert_eq!(
            rowid.encode_row_ids(&unordered, &mut output),
            Err(IdentityError::RowOrder)
        );
        assert_eq!(output, [0x5a; 56]);
        clustered.encode_row_ids(&unordered, &mut output).unwrap();
        assert!(clustered
            .decode_row_ids(&output)
            .unwrap()
            .iter()
            .eq(unordered));
        assert_eq!(
            rowid.encode_row_ids(&[i64::MIN, 0, 0], &mut output),
            Err(IdentityError::RowOrder)
        );
        assert_eq!(
            rowid.encode_row_ids(&[-1, 0, 1], &mut output),
            Err(IdentityError::RowBounds)
        );
        rowid
            .encode_row_ids(&[i64::MIN, 0, i64::MAX], &mut output)
            .unwrap();
        for (offset, value, error) in [
            (0, 0, IdentityError::Magic),
            (4, 2, IdentityError::Revision),
            (6, 31, IdentityError::Length),
            (8, 1, IdentityError::Identity),
            (16, 1, IdentityError::Identity),
            (24, 2, IdentityError::Identity),
            (28, 1, IdentityError::Layout),
            (29, 4, IdentityError::LaneWidth),
            (30, 1, IdentityError::Reserved),
        ] {
            let mut bytes = output;
            bytes[offset] = value;
            assert_eq!(rowid.decode_row_ids(&bytes).unwrap_err(), error);
        }
        let mut duplicate = output;
        duplicate[40..48].copy_from_slice(&i64::MIN.to_le_bytes());
        assert_eq!(
            rowid.decode_row_ids(&duplicate).unwrap_err(),
            IdentityError::RowOrder
        );
        let mut bad_extrema = output;
        bad_extrema[48..56].copy_from_slice(&1i64.to_le_bytes());
        assert_eq!(
            rowid.decode_row_ids(&bad_extrema).unwrap_err(),
            IdentityError::RowBounds
        );
    }
    #[test]
    fn identity_writer_requires_matching_root_feature_before_any_root_bytes() {
        use super::super::page_io::PageWriter;
        let (original, _) = expected(Layout::RowId);
        let (header, mut root, _, _) = base_context(original, 0);
        root.row_count = 0;
        root.group_count = 0;
        root.entry_count = 0;
        root.rows = None;
        root.directory = None;
        let limits = ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: 4096,
            page_decoded_bytes: 4096,
        };
        let mut sink = Vec::new();
        let mut writer = PageWriter::new(&mut sink, Header::new(header.identity), limits).unwrap();
        let before = writer.position();
        assert!(matches!(
            writer.finish(&root),
            Err(PageIoError::Directory(DirectoryError::LegacyBase))
        ));
        assert_eq!(writer.position(), before);
        writer
            .finish(&RootSummary {
                legacy_base: None,
                ..root
            })
            .unwrap();
        let mut sink = Vec::new();
        let mut writer = PageWriter::new(&mut sink, header, limits).unwrap();
        let result = writer.finish(&root).unwrap();
        let decoded = RootSummary::decode(&sink[64..192], &result.footer, &limits).unwrap();
        assert_eq!(decoded.legacy_base, root.legacy_base);
        // Writing/decoding those bytes still did not establish installer proof.
        assert!(matches!(
            SourceContext::bind(&header, &decoded, None),
            Err(IdentityError::MissingEvidence)
        ));
    }

    #[test]
    fn identity_read_plan_exact_limit_before_buffer_reservation() {
        let (_, expected) = expected(Layout::RowId);
        let raw = PageDescriptor {
            offset: 64,
            stored_len: 56,
            decoded_len: 56,
            stored_checksum: 0,
            codec: Codec::Raw,
        };
        let footer = Footer {
            file_length: 4096 + 192,
            root: PageDescriptor {
                offset: 4096,
                stored_len: 128,
                decoded_len: 128,
                ..raw
            },
        };
        let limits = ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: 4096,
            page_decoded_bytes: MAX_IDENTITY_PAGE_BYTES as u64,
        };
        for decoded_len in [0, 32, 55, 57, MAX_IDENTITY_PAGE_BYTES as u64, u64::MAX] {
            assert!(matches!(
                expected.read_plan(&footer, PageDescriptor { decoded_len, ..raw }, &limits),
                Err(PageIoError::PayloadLength)
            ));
        }
        let plan = expected.read_plan(&footer, raw, &limits).unwrap();
        assert_eq!(
            (plan.stored_buffer_len(), plan.decoded_buffer_len()),
            (56, 0)
        );
        let compressed = PageDescriptor {
            stored_len: 20,
            codec: Codec::Lz4Block,
            ..raw
        };
        let plan = expected.read_plan(&footer, compressed, &limits).unwrap();
        assert_eq!(
            (plan.stored_buffer_len(), plan.decoded_buffer_len()),
            (20, 56)
        );
        let bounds = RowBounds { min: 0, max: 4095 };
        let root = root(Layout::RowId, 4096, 1, bounds);
        let full = IdentityPageExpectation::new(
            GroupExpectation::new(
                &root,
                0,
                GroupRecord {
                    row_start: 0,
                    row_count: 4096,
                    column_count: 2,
                    rows: bounds,
                },
            )
            .unwrap(),
        );
        assert_eq!(full.decoded_len(), MAX_IDENTITY_PAGE_BYTES);
        let ids = std::array::from_fn::<_, 4096, _>(|i| i as i64);
        let mut output = [0; MAX_IDENTITY_PAGE_BYTES];
        full.encode_row_ids(&ids, &mut output).unwrap();
        assert_eq!(full.decode_row_ids(&output).unwrap().iter().count(), 4096);
    }
}
