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

//! Bounded V5 directory node and root-summary codecs. No codec owns a Vec or
//! performs I/O. Inputs must already have passed their stored-byte checksum and
//! exact-length decompression checks. Outputs are fixed arrays or caller buffers.
//!
//! Node header (32 bytes, little endian): `V5DN[0..4], revision:u16[4..6],
//! kind:u8[6], depth:u8[7], count:u16[8..10], entry_size:u16[10..12],
//! reserved_zero[12..16], subtree_entries:u64[16..24], reserved_zero[24..32]`.
//! Kind 0 is a leaf (depth 1); kind 1 is an interior node (depth 2..8). Nodes
//! contain 1..64 entries and consume their exact declared length, at most 8 KiB.
//! Leaf entries are 48 bytes: key16 + descriptor32. Interior entries are 64
//! bytes: inclusive lower key16 + inclusive upper key16 + child descriptor32.
//!
//! Key16: section:u16, flags:u16, column:u32, ordinal:u64. Flags must be exactly
//! REQUIRED (1) or OPTIONAL (2). Identity ordering is (section,column,ordinal),
//! excluding flags: changing optionality cannot create a second copy of a key.
//! u32::MAX denotes global sections. Unknown required sections fail; unknown
//! optional sections remain represented and may be skipped, never used to prove
//! absence. Payload readers enforce each known section's column/group semantics.
//!
//! Root summary (128 bytes): `V5RS[0..4], revision:u16[4..6], size:u16[6..8],
//! presence_flags:u32[8..12], layout:u8[12], directory_depth:u8[13],
//! reserved_zero[14..16], rows:u64[16..24], columns:u32[24..28],
//! reserved_zero[28..32], groups:u64[32..40], entries:u64[40..48],
//! min_row_id:i64[48..56], max_row_id:i64[56..64], window_lower:i64[64..72],
//! window_upper:i64[72..80], directory_root_descriptor[80..112],
//! legacy_generation:u64[112..120], legacy_barrier_lsn:u64[120..128]`.
//! Presence bits 0/1/2/3 mean row extrema/window/directory/LegacyBase. LegacyBase
//! requires nonzero generation E (barrier G may be zero) and the matching required
//! envelope feature. This metadata is not proof of a durable upgrade checkpoint;
//! callers must bind installer evidence before interpreting source lane zero.
//! Absent fields are zero. Window bounds are closed extrema of normalized window
//! IDs (equal bounds denote one window), not timestamps or half-open time
//! intervals. Timestamp-to-window conversion belongs to the clustering policy.
//! Layout 0 is RowId and 1 is Clustered. Empty volumes still have this nonempty
//! summary.
//!
//! Directory nodes are emitted in increasing physical postorder: each subtree
//! precedes its parent and earlier key subtrees precede later key subtrees.
//! Direct sibling child-root extents are strictly ascending and nonoverlapping.
//! The streaming structural walker checks complete postorder extents, including
//! descendants across sibling subtrees, without retaining an extent collection.
//! Directory pages are appended after their payloads/children. Every reference
//! ends at or before its referring node's offset; strict backward addresses and
//! depth <=8 give a bounded acyclic traversal. Sibling key ranges are disjoint.
//! On descent, validate_child checks exact child range, identity and depth.
//! Whole-tree totals and directory-page overlap are checked by the structural
//! walker. Payload extents and logical group/payload coverage require their
//! subsequent payload validators; local codecs do not claim these properties.

use std::cmp::Ordering;
use std::fmt;
use std::num::NonZeroU64;

use super::envelope::{
    EnvelopeError, Footer, Header, LegacyBase, PageDescriptor, ReadLimits, REQUIRED_LEGACY_BASE,
};

pub const MAX_FANOUT: usize = 64;
pub const MAX_DEPTH: u8 = 8;
pub const MAX_NODE_BYTES: usize = 8192;
pub const NODE_HEADER_BYTES: usize = 32;
pub const ROOT_SUMMARY_BYTES: usize = 128;
pub const KEY_BYTES: usize = 16;
pub const LEAF_ENTRY_BYTES: usize = 48;
pub const INTERIOR_ENTRY_BYTES: usize = 64;
pub const KEY_REQUIRED: u16 = 1;
pub const KEY_OPTIONAL: u16 = 2;
pub const GLOBAL_COLUMN: u32 = u32::MAX;
const REVISION: u16 = 1;
const HAS_ROWS: u32 = 1;
const HAS_WINDOW: u32 = 2;
const HAS_DIRECTORY: u32 = 4;
const HAS_LEGACY_BASE: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Section {
    ColumnBlocks = 1,
    RowIds = 2,
    SourceLsns = 3,
    LocatorRowIds = 4,
    LocatorOrdinals = 5,
    Dictionaries = 6,
    GroupMetadata = 7,
    RunMetadata = 8,
    Blooms = 9,
    ConstraintPages = 10,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectoryError {
    Envelope(EnvelopeError),
    Length,
    Magic,
    Revision,
    Reserved,
    NodeKind,
    Depth,
    Fanout,
    EntrySize,
    EntryCount,
    KeyFlags,
    UnknownRequiredSection(u16),
    KeyOrder,
    PhysicalOrder,
    ForwardReference,
    ChildMismatch,
    Layout,
    PresenceFlags,
    RootState,
    RowBounds,
    WindowBounds,
    OutputTooShort,
    LegacyBase,
}
impl fmt::Display for DirectoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Envelope(error) => write!(f, "V5 directory: {error}"),
            other => write!(f, "invalid V5 directory: {other:?}"),
        }
    }
}
impl std::error::Error for DirectoryError {}
impl From<EnvelopeError> for DirectoryError {
    fn from(error: EnvelopeError) -> Self {
        Self::Envelope(error)
    }
}
type Result<T> = std::result::Result<T, DirectoryError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectoryKey {
    pub section: u16,
    pub flags: u16,
    pub column: u32,
    pub ordinal: u64,
}
impl DirectoryKey {
    /// Key identity deliberately excludes required/optional flags.
    pub const fn identity(&self) -> (u16, u32, u64) {
        (self.section, self.column, self.ordinal)
    }
    pub fn cmp_identity(&self, other: &Self) -> Ordering {
        self.identity().cmp(&other.identity())
    }
    pub const fn is_known_section(&self) -> bool {
        self.section >= 1 && self.section <= 10
    }
    pub const fn skip_unknown_optional(&self) -> bool {
        !self.is_known_section() && self.flags == KEY_OPTIONAL
    }
    pub(crate) fn validate(&self) -> Result<()> {
        if !matches!(self.flags, KEY_REQUIRED | KEY_OPTIONAL) {
            return Err(DirectoryError::KeyFlags);
        }
        if !self.is_known_section() && self.flags == KEY_REQUIRED {
            return Err(DirectoryError::UnknownRequiredSection(self.section));
        }
        Ok(())
    }
    fn decode(bytes: &[u8]) -> Result<Self> {
        let key = Self {
            section: u16_at(bytes, 0),
            flags: u16_at(bytes, 2),
            column: u32_at(bytes, 4),
            ordinal: u64_at(bytes, 8),
        };
        key.validate()?;
        Ok(key)
    }
    fn encode(&self, bytes: &mut [u8]) {
        put_u16(bytes, 0, self.section);
        put_u16(bytes, 2, self.flags);
        put_u32(bytes, 4, self.column);
        put_u64(bytes, 8, self.ordinal);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LeafEntry {
    pub key: DirectoryKey,
    pub page: PageDescriptor,
}
impl LeafEntry {
    /// Constant-size decode for a cursor over an already validated node buffer.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != LEAF_ENTRY_BYTES {
            return Err(DirectoryError::Length);
        }
        leaf_at(bytes)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteriorEntry {
    pub lower: DirectoryKey,
    pub upper: DirectoryKey,
    pub child: PageDescriptor,
}
impl InteriorEntry {
    /// Constant-size decode for a cursor over an already validated node buffer.
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != INTERIOR_ENTRY_BYTES {
            return Err(DirectoryError::Length);
        }
        interior_at(bytes)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Leaf,
    Interior,
}

/// Borrowed, locally validated node. Lifetime pins only the caller's decoded
/// byte slice; no index or entry vector is reconstructed.
#[derive(Clone, Copy, Debug)]
pub struct DirectoryNode<'a> {
    bytes: &'a [u8],
    location: PageDescriptor,
    kind: NodeKind,
    depth: u8,
    count: u16,
    subtree_entries: u64,
}
impl<'a> DirectoryNode<'a> {
    pub fn decode(
        bytes: &'a [u8],
        location: PageDescriptor,
        footer: &Footer,
        limits: &ReadLimits,
    ) -> Result<Self> {
        footer.validate_page(&location, limits)?;
        if bytes.len() < NODE_HEADER_BYTES
            || bytes.len() > MAX_NODE_BYTES
            || u64::try_from(bytes.len()).ok() != Some(location.decoded_len)
        {
            return Err(DirectoryError::Length);
        }
        if &bytes[..4] != b"V5DN" {
            return Err(DirectoryError::Magic);
        }
        if u16_at(bytes, 4) != REVISION {
            return Err(DirectoryError::Revision);
        }
        require_zero(&bytes[12..16])?;
        require_zero(&bytes[24..32])?;
        let kind = match bytes[6] {
            0 => NodeKind::Leaf,
            1 => NodeKind::Interior,
            _ => return Err(DirectoryError::NodeKind),
        };
        let depth = bytes[7];
        validate_depth(kind, depth)?;
        let count = u16_at(bytes, 8);
        let size = entry_size(kind);
        if usize::from(u16_at(bytes, 10)) != size {
            return Err(DirectoryError::EntrySize);
        }
        if node_length(usize::from(count), size)? != bytes.len() {
            return Err(DirectoryError::Length);
        }
        let subtree_entries = u64_at(bytes, 16);
        validate_subtree(kind, count, subtree_entries)?;
        let node = Self {
            bytes,
            location,
            kind,
            depth,
            count,
            subtree_entries,
        };
        let mut previous = None;
        match kind {
            NodeKind::Leaf => {
                for entry in node.leaves() {
                    let entry = entry?;
                    validate_order(previous, entry.key)?;
                    footer.validate_page(&entry.page, limits)?;
                    validate_backward(&entry.page, location.offset)?;
                    previous = Some(entry.key);
                }
            }
            NodeKind::Interior => {
                let mut previous_end = None;
                for entry in node.children() {
                    let entry = entry?;
                    if entry.lower.cmp_identity(&entry.upper) == Ordering::Greater {
                        return Err(DirectoryError::KeyOrder);
                    }
                    validate_order(previous, entry.lower)?;
                    footer.validate_page(&entry.child, limits)?;
                    validate_backward(&entry.child, location.offset)?;
                    validate_physical_order(previous_end, entry.child)?;
                    previous_end = Some(page_end(entry.child)?);
                    previous = Some(entry.upper);
                }
            }
        }
        Ok(node)
    }
    pub const fn kind(&self) -> NodeKind {
        self.kind
    }
    pub const fn depth(&self) -> u8 {
        self.depth
    }
    pub const fn len(&self) -> usize {
        self.count as usize
    }
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
    pub const fn subtree_entries(&self) -> u64 {
        self.subtree_entries
    }
    pub fn leaves(&self) -> LeafEntries<'a> {
        LeafEntries {
            bytes: if self.kind == NodeKind::Leaf {
                &self.bytes[NODE_HEADER_BYTES..]
            } else {
                &[]
            },
        }
    }
    pub fn children(&self) -> InteriorEntries<'a> {
        InteriorEntries {
            bytes: if self.kind == NodeKind::Interior {
                &self.bytes[NODE_HEADER_BYTES..]
            } else {
                &[]
            },
        }
    }
    pub fn key_range(&self) -> (DirectoryKey, DirectoryKey) {
        // The complete node was validated before construction; all counts are
        // positive and every entry parser already succeeded.
        match self.kind {
            NodeKind::Leaf => {
                let first = leaf_at(&self.bytes[NODE_HEADER_BYTES..]).expect("validated leaf");
                let last = leaf_at(&self.bytes[self.bytes.len() - LEAF_ENTRY_BYTES..])
                    .expect("validated leaf");
                (first.key, last.key)
            }
            NodeKind::Interior => {
                let first =
                    interior_at(&self.bytes[NODE_HEADER_BYTES..]).expect("validated interior");
                let last = interior_at(&self.bytes[self.bytes.len() - INTERIOR_ENTRY_BYTES..])
                    .expect("validated interior");
                (first.lower, last.upper)
            }
        }
    }
    pub fn validate_child(&self, ordinal: usize, child: &DirectoryNode<'_>) -> Result<()> {
        if self.kind != NodeKind::Interior {
            return Err(DirectoryError::NodeKind);
        }
        let entry = self
            .children()
            .nth(ordinal)
            .ok_or(DirectoryError::ChildMismatch)??;
        let (lower, upper) = child.key_range();
        if entry.child != child.location
            || child.depth + 1 != self.depth
            || entry.lower.identity() != lower.identity()
            || entry.upper.identity() != upper.identity()
        {
            return Err(DirectoryError::ChildMismatch);
        }
        Ok(())
    }
}

pub struct LeafEntries<'a> {
    bytes: &'a [u8],
}
impl Iterator for LeafEntries<'_> {
    type Item = Result<LeafEntry>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes.is_empty() {
            return None;
        }
        let (entry, tail) = self.bytes.split_at(LEAF_ENTRY_BYTES);
        self.bytes = tail;
        Some(leaf_at(entry))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.bytes.len() / LEAF_ENTRY_BYTES;
        (n, Some(n))
    }
}
impl ExactSizeIterator for LeafEntries<'_> {}
impl std::iter::FusedIterator for LeafEntries<'_> {}
pub struct InteriorEntries<'a> {
    bytes: &'a [u8],
}
impl Iterator for InteriorEntries<'_> {
    type Item = Result<InteriorEntry>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes.is_empty() {
            return None;
        }
        let (entry, tail) = self.bytes.split_at(INTERIOR_ENTRY_BYTES);
        self.bytes = tail;
        Some(interior_at(entry))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.bytes.len() / INTERIOR_ENTRY_BYTES;
        (n, Some(n))
    }
}
impl ExactSizeIterator for InteriorEntries<'_> {}
impl std::iter::FusedIterator for InteriorEntries<'_> {}

/// Encode into caller storage. Since the final node offset is not yet known,
/// the writer must validate descriptor bounds/backward references once placed.
/// Validation errors leave the complete output buffer unchanged.
pub fn encode_leaf(entries: &[LeafEntry], output: &mut [u8]) -> Result<usize> {
    let length = node_length(entries.len(), LEAF_ENTRY_BYTES)?;
    if output.len() < length {
        return Err(DirectoryError::OutputTooShort);
    }
    let mut previous = None;
    for entry in entries {
        entry.key.validate()?;
        validate_order(previous, entry.key)?;
        entry.page.encode()?;
        previous = Some(entry.key);
    }
    encode_node_header(
        output,
        NodeKind::Leaf,
        1,
        entries.len(),
        entries.len() as u64,
    );
    for (entry, bytes) in entries.iter().zip(
        output[NODE_HEADER_BYTES..length]
            .as_chunks_mut::<LEAF_ENTRY_BYTES>()
            .0,
    ) {
        entry.key.encode(bytes);
        bytes[KEY_BYTES..].copy_from_slice(&entry.page.encode()?);
    }
    Ok(length)
}
/// See [`encode_leaf`] for placement validation and output ownership.
pub fn encode_interior(
    depth: u8,
    subtree_entries: u64,
    entries: &[InteriorEntry],
    output: &mut [u8],
) -> Result<usize> {
    validate_depth(NodeKind::Interior, depth)?;
    let length = node_length(entries.len(), INTERIOR_ENTRY_BYTES)?;
    validate_subtree(NodeKind::Interior, entries.len() as u16, subtree_entries)?;
    if output.len() < length {
        return Err(DirectoryError::OutputTooShort);
    }
    let mut previous = None;
    let mut previous_end = None;
    for entry in entries {
        entry.lower.validate()?;
        entry.upper.validate()?;
        if entry.lower.cmp_identity(&entry.upper) == Ordering::Greater {
            return Err(DirectoryError::KeyOrder);
        }
        validate_order(previous, entry.lower)?;
        entry.child.encode()?;
        validate_physical_order(previous_end, entry.child)?;
        previous_end = Some(page_end(entry.child)?);
        previous = Some(entry.upper);
    }
    encode_node_header(
        output,
        NodeKind::Interior,
        depth,
        entries.len(),
        subtree_entries,
    );
    for (entry, bytes) in entries.iter().zip(
        output[NODE_HEADER_BYTES..length]
            .as_chunks_mut::<INTERIOR_ENTRY_BYTES>()
            .0,
    ) {
        entry.lower.encode(bytes);
        entry.upper.encode(&mut bytes[KEY_BYTES..]);
        bytes[2 * KEY_BYTES..].copy_from_slice(&entry.child.encode()?);
    }
    Ok(length)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    RowId,
    Clustered,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowBounds {
    pub min: i64,
    pub max: i64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Inclusive extrema of normalized window IDs; a single window has equal bounds.
pub struct WindowBounds {
    pub lower: i64,
    pub upper: i64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectoryRoot {
    pub depth: u8,
    pub page: PageDescriptor,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RootSummary {
    pub layout: Layout,
    pub row_count: u64,
    pub column_count: u32,
    pub group_count: u64,
    pub entry_count: u64,
    pub rows: Option<RowBounds>,
    pub window: Option<WindowBounds>,
    pub directory: Option<DirectoryRoot>,
    /// Untrusted checkpoint metadata until paired with verified installer evidence.
    pub legacy_base: Option<LegacyBase>,
}
impl RootSummary {
    pub fn encode(&self) -> Result<[u8; ROOT_SUMMARY_BYTES]> {
        self.validate()?;
        let mut bytes = [0; ROOT_SUMMARY_BYTES];
        bytes[..4].copy_from_slice(b"V5RS");
        put_u16(&mut bytes, 4, REVISION);
        put_u16(&mut bytes, 6, ROOT_SUMMARY_BYTES as u16);
        let flags = (u32::from(self.rows.is_some()) * HAS_ROWS)
            | (u32::from(self.window.is_some()) * HAS_WINDOW)
            | (u32::from(self.directory.is_some()) * HAS_DIRECTORY)
            | (u32::from(self.legacy_base.is_some()) * HAS_LEGACY_BASE);
        put_u32(&mut bytes, 8, flags);
        bytes[12] = match self.layout {
            Layout::RowId => 0,
            Layout::Clustered => 1,
        };
        put_u64(&mut bytes, 16, self.row_count);
        put_u32(&mut bytes, 24, self.column_count);
        put_u64(&mut bytes, 32, self.group_count);
        put_u64(&mut bytes, 40, self.entry_count);
        if let Some(rows) = self.rows {
            put_i64(&mut bytes, 48, rows.min);
            put_i64(&mut bytes, 56, rows.max);
        }
        if let Some(window) = self.window {
            put_i64(&mut bytes, 64, window.lower);
            put_i64(&mut bytes, 72, window.upper);
        }
        if let Some(root) = self.directory {
            bytes[13] = root.depth;
            bytes[80..112].copy_from_slice(&root.page.encode()?);
        }
        if let Some(base) = self.legacy_base {
            put_u64(&mut bytes, 112, base.generation.get());
            put_u64(&mut bytes, 120, base.barrier_lsn);
        }
        Ok(bytes)
    }
    pub fn decode(bytes: &[u8], footer: &Footer, limits: &ReadLimits) -> Result<Self> {
        footer.validate_root(limits)?;
        if bytes.len() != ROOT_SUMMARY_BYTES || footer.root.decoded_len != ROOT_SUMMARY_BYTES as u64
        {
            return Err(DirectoryError::Length);
        }
        if &bytes[..4] != b"V5RS" {
            return Err(DirectoryError::Magic);
        }
        if u16_at(bytes, 4) != REVISION {
            return Err(DirectoryError::Revision);
        }
        if u16_at(bytes, 6) as usize != ROOT_SUMMARY_BYTES {
            return Err(DirectoryError::Length);
        }
        require_zero(&bytes[14..16])?;
        require_zero(&bytes[28..32])?;
        let flags = u32_at(bytes, 8);
        if flags & !(HAS_ROWS | HAS_WINDOW | HAS_DIRECTORY | HAS_LEGACY_BASE) != 0 {
            return Err(DirectoryError::PresenceFlags);
        }
        let layout = match bytes[12] {
            0 => Layout::RowId,
            1 => Layout::Clustered,
            _ => return Err(DirectoryError::Layout),
        };
        let rows = if flags & HAS_ROWS != 0 {
            Some(RowBounds {
                min: i64_at(bytes, 48),
                max: i64_at(bytes, 56),
            })
        } else {
            require_zero(&bytes[48..64])?;
            None
        };
        let window = if flags & HAS_WINDOW != 0 {
            Some(WindowBounds {
                lower: i64_at(bytes, 64),
                upper: i64_at(bytes, 72),
            })
        } else {
            require_zero(&bytes[64..80])?;
            None
        };
        let directory = if flags & HAS_DIRECTORY != 0 {
            Some(DirectoryRoot {
                depth: bytes[13],
                page: PageDescriptor::decode(&bytes[80..112])?,
            })
        } else {
            if bytes[13] != 0 {
                return Err(DirectoryError::RootState);
            }
            require_zero(&bytes[80..112])?;
            None
        };
        let legacy_base = if flags & HAS_LEGACY_BASE != 0 {
            Some(LegacyBase {
                generation: NonZeroU64::new(u64_at(bytes, 112))
                    .ok_or(DirectoryError::LegacyBase)?,
                barrier_lsn: u64_at(bytes, 120),
            })
        } else {
            require_zero(&bytes[112..128])?;
            None
        };
        let root = Self {
            layout,
            row_count: u64_at(bytes, 16),
            column_count: u32_at(bytes, 24),
            group_count: u64_at(bytes, 32),
            entry_count: u64_at(bytes, 40),
            rows,
            window,
            directory,
            legacy_base,
        };
        root.validate()?;
        if let Some(directory) = root.directory {
            footer.validate_page(&directory.page, limits)?;
        }
        Ok(root)
    }
    /// Structural consistency only. This does not certify installer evidence.
    /// Call after decoding the root and before interpreting any source payload.
    pub fn validate_header(&self, header: &Header) -> Result<()> {
        self.validate()?;
        header.validate_features()?;
        if self.legacy_base.is_some() != (header.required_features & REQUIRED_LEGACY_BASE != 0) {
            return Err(DirectoryError::LegacyBase);
        }
        Ok(())
    }
    pub fn validate_directory_root(&self, node: &DirectoryNode<'_>) -> Result<()> {
        self.validate()?;
        let expected = self.directory.ok_or(DirectoryError::RootState)?;
        if node.location != expected.page
            || node.depth != expected.depth
            || node.subtree_entries != self.entry_count
        {
            return Err(DirectoryError::ChildMismatch);
        }
        Ok(())
    }
    pub(crate) fn validate(&self) -> Result<()> {
        match self.rows {
            None if self.row_count == 0 && self.group_count == 0 => (),
            Some(bounds)
                if self.row_count != 0
                    && self.group_count != 0
                    && self.group_count <= self.row_count =>
            {
                if bounds.min > bounds.max
                    || u128::from(self.row_count)
                        > (i128::from(bounds.max) - i128::from(bounds.min) + 1) as u128
                {
                    return Err(DirectoryError::RowBounds);
                }
            }
            _ => return Err(DirectoryError::RootState),
        }
        if self.window.is_some_and(|w| w.lower > w.upper) {
            return Err(DirectoryError::WindowBounds);
        }
        match self.directory {
            None if self.entry_count == 0 && self.row_count == 0 => (),
            Some(root) if self.entry_count != 0 && (1..=MAX_DEPTH).contains(&root.depth) => {
                root.page.encode()?;
            }
            _ => return Err(DirectoryError::RootState),
        }
        Ok(())
    }
}

fn leaf_at(bytes: &[u8]) -> Result<LeafEntry> {
    Ok(LeafEntry {
        key: DirectoryKey::decode(&bytes[..KEY_BYTES])?,
        page: PageDescriptor::decode(&bytes[KEY_BYTES..LEAF_ENTRY_BYTES])?,
    })
}
fn interior_at(bytes: &[u8]) -> Result<InteriorEntry> {
    Ok(InteriorEntry {
        lower: DirectoryKey::decode(&bytes[..KEY_BYTES])?,
        upper: DirectoryKey::decode(&bytes[KEY_BYTES..2 * KEY_BYTES])?,
        child: PageDescriptor::decode(&bytes[2 * KEY_BYTES..INTERIOR_ENTRY_BYTES])?,
    })
}
fn validate_order(previous: Option<DirectoryKey>, key: DirectoryKey) -> Result<()> {
    if previous.is_some_and(|previous| previous.cmp_identity(&key) != Ordering::Less) {
        Err(DirectoryError::KeyOrder)
    } else {
        Ok(())
    }
}
fn validate_backward(page: &PageDescriptor, parent_offset: u64) -> Result<()> {
    if page
        .offset
        .checked_add(page.stored_len)
        .ok_or(EnvelopeError::OffsetOverflow)?
        > parent_offset
    {
        Err(DirectoryError::ForwardReference)
    } else {
        Ok(())
    }
}
fn page_end(page: PageDescriptor) -> Result<u64> {
    page.offset
        .checked_add(page.stored_len)
        .ok_or(EnvelopeError::OffsetOverflow.into())
}
fn validate_physical_order(previous_end: Option<u64>, page: PageDescriptor) -> Result<()> {
    if previous_end.is_some_and(|end| end > page.offset) {
        Err(DirectoryError::PhysicalOrder)
    } else {
        Ok(())
    }
}
fn validate_depth(kind: NodeKind, depth: u8) -> Result<()> {
    if match kind {
        NodeKind::Leaf => depth != 1,
        NodeKind::Interior => !(2..=MAX_DEPTH).contains(&depth),
    } {
        Err(DirectoryError::Depth)
    } else {
        Ok(())
    }
}
fn validate_subtree(kind: NodeKind, count: u16, total: u64) -> Result<()> {
    if match kind {
        NodeKind::Leaf => total != u64::from(count),
        NodeKind::Interior => total < u64::from(count),
    } {
        Err(DirectoryError::EntryCount)
    } else {
        Ok(())
    }
}
fn entry_size(kind: NodeKind) -> usize {
    match kind {
        NodeKind::Leaf => LEAF_ENTRY_BYTES,
        NodeKind::Interior => INTERIOR_ENTRY_BYTES,
    }
}
fn node_length(count: usize, size: usize) -> Result<usize> {
    if !(1..=MAX_FANOUT).contains(&count) {
        return Err(DirectoryError::Fanout);
    }
    NODE_HEADER_BYTES
        .checked_add(count.checked_mul(size).ok_or(DirectoryError::Length)?)
        .filter(|&len| len <= MAX_NODE_BYTES)
        .ok_or(DirectoryError::Length)
}
fn encode_node_header(bytes: &mut [u8], kind: NodeKind, depth: u8, count: usize, total: u64) {
    bytes[..NODE_HEADER_BYTES].fill(0);
    bytes[..4].copy_from_slice(b"V5DN");
    put_u16(bytes, 4, REVISION);
    bytes[6] = match kind {
        NodeKind::Leaf => 0,
        NodeKind::Interior => 1,
    };
    bytes[7] = depth;
    put_u16(bytes, 8, count as u16);
    put_u16(bytes, 10, entry_size(kind) as u16);
    put_u64(bytes, 16, total);
}
fn require_zero(bytes: &[u8]) -> Result<()> {
    if bytes.iter().any(|&b| b != 0) {
        Err(DirectoryError::Reserved)
    } else {
        Ok(())
    }
}
fn put_u16(b: &mut [u8], p: usize, v: u16) {
    b[p..p + 2].copy_from_slice(&v.to_le_bytes());
}
fn put_u32(b: &mut [u8], p: usize, v: u32) {
    b[p..p + 4].copy_from_slice(&v.to_le_bytes());
}
fn put_u64(b: &mut [u8], p: usize, v: u64) {
    b[p..p + 8].copy_from_slice(&v.to_le_bytes());
}
fn put_i64(b: &mut [u8], p: usize, v: i64) {
    b[p..p + 8].copy_from_slice(&v.to_le_bytes());
}
fn u16_at(b: &[u8], p: usize) -> u16 {
    u16::from_le_bytes(b[p..p + 2].try_into().expect("checked directory length"))
}
fn u32_at(b: &[u8], p: usize) -> u32 {
    u32::from_le_bytes(b[p..p + 4].try_into().expect("checked directory length"))
}
fn u64_at(b: &[u8], p: usize) -> u64 {
    u64::from_le_bytes(b[p..p + 8].try_into().expect("checked directory length"))
}
fn i64_at(b: &[u8], p: usize) -> i64 {
    i64::from_le_bytes(b[p..p + 8].try_into().expect("checked directory length"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::volume::v5::envelope::Codec;

    fn limits() -> ReadLimits {
        ReadLimits {
            root_stored_bytes: 128,
            root_decoded_bytes: 128,
            page_stored_bytes: MAX_NODE_BYTES as u64,
            page_decoded_bytes: MAX_NODE_BYTES as u64,
        }
    }
    fn page(offset: u64, len: u64) -> PageDescriptor {
        PageDescriptor {
            offset,
            stored_len: len,
            decoded_len: len,
            stored_checksum: 0,
            codec: Codec::Raw,
        }
    }
    fn footer() -> Footer {
        Footer {
            file_length: 32_768 + 128 + 64,
            root: page(32_768, 128),
        }
    }
    fn key(ordinal: u64) -> DirectoryKey {
        DirectoryKey {
            section: Section::SourceLsns as u16,
            flags: KEY_REQUIRED,
            column: GLOBAL_COLUMN,
            ordinal,
        }
    }
    fn leaf(ordinal: u64) -> LeafEntry {
        LeafEntry {
            key: key(ordinal),
            page: page(64, 8),
        }
    }
    fn decode<'a>(bytes: &'a [u8], offset: u64) -> Result<DirectoryNode<'a>> {
        DirectoryNode::decode(
            bytes,
            page(offset, bytes.len() as u64),
            &footer(),
            &limits(),
        )
    }
    fn fixed_hex<const N: usize>(text: &str) -> [u8; N] {
        assert_eq!(text.len(), N * 2);
        std::array::from_fn(|i| u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).unwrap())
    }
    fn root() -> RootSummary {
        RootSummary {
            layout: Layout::Clustered,
            legacy_base: None,
            row_count: 2,
            column_count: 3,
            group_count: 1,
            entry_count: 1,
            rows: Some(RowBounds { min: -2, max: 9 }),
            window: Some(WindowBounds {
                lower: -100,
                upper: 100,
            }),
            directory: Some(DirectoryRoot {
                depth: 1,
                page: PageDescriptor {
                    stored_checksum: 0xabcd_ef01,
                    ..page(256, 80)
                },
            }),
        }
    }

    #[test]
    fn legacy_root_wire_requires_feature_and_preserves_untrusted_metadata() {
        // Independently generated Python struct/zlib vectors, E nonzero and G=0.
        let golden = fixed_hex::<128>("5635525301008000080000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000008070605040302010000000000000000");
        let header_golden = fixed_hex::<64>("5354563501004000070000000000000000000000000000000b000000000000000100000000000000160000000000000000000000000000000000000070d5587d");
        assert_eq!(crc32fast::hash(&golden), 0xf2ca_d2d7);
        let base = LegacyBase {
            generation: NonZeroU64::new(0x0102_0304_0506_0708).unwrap(),
            barrier_lsn: 0,
        };
        let root = RootSummary {
            layout: Layout::RowId,
            row_count: 0,
            column_count: 0,
            group_count: 0,
            entry_count: 0,
            rows: None,
            window: None,
            directory: None,
            legacy_base: Some(base),
        };
        assert_eq!(root.encode().unwrap(), golden);
        let decoded = RootSummary::decode(&golden, &footer(), &limits()).unwrap();
        assert_eq!(decoded, root);
        let header = Header::decode(&header_golden).unwrap();
        assert_eq!(header.encode().unwrap(), header_golden);
        decoded.validate_header(&header).unwrap();
        assert_eq!(
            decoded.validate_header(&Header::new(header.identity)),
            Err(DirectoryError::LegacyBase)
        );
        assert_eq!(
            RootSummary {
                legacy_base: None,
                ..decoded
            }
            .validate_header(&header),
            Err(DirectoryError::LegacyBase)
        );
        let mut zero_e = golden;
        zero_e[112..120].fill(0);
        assert_eq!(
            RootSummary::decode(&zero_e, &footer(), &limits()),
            Err(DirectoryError::LegacyBase)
        );
        let mut absent = golden;
        absent[8] = 0;
        assert_eq!(
            RootSummary::decode(&absent, &footer(), &limits()),
            Err(DirectoryError::Reserved)
        );
        for barrier_lsn in [0, 1, u64::MAX] {
            let root = RootSummary {
                legacy_base: Some(LegacyBase {
                    barrier_lsn,
                    ..base
                }),
                ..root
            };
            assert_eq!(
                RootSummary::decode(&root.encode().unwrap(), &footer(), &limits()).unwrap(),
                root
            );
        }
    }

    #[test]
    fn directory_golden_leaf_and_root_bytes() {
        // Independently generated with Python struct.pack, not the Rust encoder.
        let expected_leaf = fixed_hex::<80>("5635444e0100000101003000000000000100000000000000000000000000000003000100ffffffff07000000000000004000000000000000080000000000000008000000000000007856341200000000");
        let expected_root = fixed_hex::<128>("563552530100800007000000010100000200000000000000030000000000000001000000000000000100000000000000feffffffffffffff09000000000000009cffffffffffffff640000000000000000010000000000005000000000000000500000000000000001efcdab0000000000000000000000000000000000000000");
        let entry = LeafEntry {
            page: PageDescriptor {
                stored_checksum: 0x1234_5678,
                ..page(64, 8)
            },
            ..leaf(7)
        };
        let mut encoded = [0x5a; 96];
        assert_eq!(encode_leaf(&[entry], &mut encoded).unwrap(), 80);
        assert_eq!(encoded[..80], expected_leaf);
        assert_eq!(encoded[80..], [0x5a; 16]);
        let decoded = decode(&expected_leaf, 256).unwrap();
        assert_eq!(decoded.leaves().next().unwrap().unwrap(), entry);
        assert_eq!(decoded.key_range(), (entry.key, entry.key));
        assert_eq!(decoded.children().len(), 0);
        assert_eq!(root().encode().unwrap(), expected_root);
        assert_eq!(
            RootSummary::decode(&expected_root, &footer(), &limits()).unwrap(),
            root()
        );
    }

    #[test]
    fn directory_full_fanout_and_borrowed_iteration() {
        let entries: [LeafEntry; MAX_FANOUT] = std::array::from_fn(|i| leaf(i as u64));
        let mut buffer = [0; MAX_NODE_BYTES];
        let len = encode_leaf(&entries, &mut buffer).unwrap();
        assert_eq!(len, NODE_HEADER_BYTES + MAX_FANOUT * LEAF_ENTRY_BYTES);
        let node = decode(&buffer[..len], 4096).unwrap();
        assert_eq!(node.len(), MAX_FANOUT);
        assert_eq!(node.subtree_entries(), MAX_FANOUT as u64);
        assert!(!node.is_empty());
        let mut it = node.leaves();
        for (i, expected) in entries.into_iter().enumerate() {
            assert_eq!(it.len(), MAX_FANOUT - i);
            assert_eq!(it.next().unwrap().unwrap(), expected);
        }
        assert!(it.next().is_none());
        assert!(it.next().is_none());
        let inner: [InteriorEntry; MAX_FANOUT] = std::array::from_fn(|i| InteriorEntry {
            lower: key(i as u64 * 2),
            upper: key(i as u64 * 2 + 1),
            child: page(256 + i as u64 * 80, 80),
        });
        let len = encode_interior(MAX_DEPTH, 128, &inner, &mut buffer).unwrap();
        assert_eq!(len, NODE_HEADER_BYTES + MAX_FANOUT * INTERIOR_ENTRY_BYTES);
        let node = decode(&buffer[..len], 8192).unwrap();
        assert_eq!(node.depth(), MAX_DEPTH);
        assert_eq!(node.children().len(), MAX_FANOUT);
        assert_eq!(node.leaves().len(), 0);
    }

    #[test]
    fn directory_rejects_lengths_headers_and_entry_counts() {
        let mut bytes = [0; 80];
        encode_leaf(&[leaf(1)], &mut bytes).unwrap();
        for len in 0..bytes.len() {
            assert!(decode(&bytes[..len], 256).is_err(), "truncation {len}");
        }
        let mut longer = [0; 81];
        longer[..80].copy_from_slice(&bytes);
        assert!(decode(&longer, 256).is_err());
        for (position, value, expected) in [
            (0, 0, DirectoryError::Magic),
            (4, 2, DirectoryError::Revision),
            (6, 2, DirectoryError::NodeKind),
            (7, 0, DirectoryError::Depth),
            (7, 2, DirectoryError::Depth),
            (8, 0, DirectoryError::Fanout),
            (8, 65, DirectoryError::Fanout),
            (10, 64, DirectoryError::EntrySize),
            (12, 1, DirectoryError::Reserved),
            (16, 0, DirectoryError::EntryCount),
            (16, 2, DirectoryError::EntryCount),
            (24, 1, DirectoryError::Reserved),
        ] {
            let mut bad = bytes;
            bad[position] = value;
            assert_eq!(decode(&bad, 256).unwrap_err(), expected, "byte {position}");
        }
        let mut too_large = [0; MAX_NODE_BYTES + 1];
        too_large[..80].copy_from_slice(&bytes);
        assert!(decode(&too_large, 10_000).is_err());
        let wrong_len = PageDescriptor {
            decoded_len: 81,
            stored_len: 81,
            ..page(256, 80)
        };
        assert_eq!(
            DirectoryNode::decode(&bytes, wrong_len, &footer(), &limits()).unwrap_err(),
            DirectoryError::Length
        );
    }

    #[test]
    fn directory_key_identity_ignores_optionality_and_preserves_unknown() {
        let mut bytes = [0; 128];
        let mut duplicate = leaf(1);
        duplicate.key.flags = KEY_OPTIONAL;
        assert_eq!(
            encode_leaf(&[leaf(1), duplicate], &mut bytes),
            Err(DirectoryError::KeyOrder)
        );
        encode_leaf(&[leaf(1), leaf(2)], &mut bytes).unwrap();
        // A checksum-valid byte stream can still contain a duplicate with different flags.
        put_u16(&mut bytes, 82, KEY_OPTIONAL);
        put_u64(&mut bytes, 88, 1);
        assert_eq!(decode(&bytes, 256).unwrap_err(), DirectoryError::KeyOrder);
        let unknown = LeafEntry {
            key: DirectoryKey {
                section: 65000,
                flags: KEY_OPTIONAL,
                ..key(9)
            },
            ..leaf(9)
        };
        let len = encode_leaf(&[unknown], &mut bytes).unwrap();
        let got = decode(&bytes[..len], 256)
            .unwrap()
            .leaves()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(got, unknown);
        assert!(got.key.skip_unknown_optional());
        put_u16(&mut bytes, 34, KEY_REQUIRED);
        assert_eq!(
            decode(&bytes[..len], 256).unwrap_err(),
            DirectoryError::UnknownRequiredSection(65000)
        );
        for flags in [0, 3, u16::MAX] {
            put_u16(&mut bytes, 34, flags);
            assert_eq!(
                decode(&bytes[..len], 256).unwrap_err(),
                DirectoryError::KeyFlags
            );
        }
    }

    #[test]
    fn directory_backward_bounds_and_child_identity_are_checked() {
        let mut child_bytes = [0; 80];
        encode_leaf(&[leaf(1)], &mut child_bytes).unwrap();
        let child_location = page(256, 80);
        let child =
            DirectoryNode::decode(&child_bytes, child_location, &footer(), &limits()).unwrap();
        let entry = InteriorEntry {
            lower: key(1),
            upper: key(1),
            child: child_location,
        };
        let mut parent_bytes = [0; 96];
        encode_interior(2, 1, &[entry], &mut parent_bytes).unwrap();
        // Exact adjacency is valid; even one byte of parent overlap is rejected.
        let parent = decode(&parent_bytes, 336).unwrap();
        parent.validate_child(0, &child).unwrap();
        assert_eq!(
            parent.validate_child(1, &child),
            Err(DirectoryError::ChildMismatch)
        );
        assert_eq!(
            decode(&parent_bytes, 335).unwrap_err(),
            DirectoryError::ForwardReference
        );
        let different_location =
            DirectoryNode::decode(&child_bytes, page(255, 80), &footer(), &limits()).unwrap();
        assert_eq!(
            parent.validate_child(0, &different_location),
            Err(DirectoryError::ChildMismatch)
        );
        encode_interior(3, 1, &[entry], &mut parent_bytes).unwrap();
        assert_eq!(
            decode(&parent_bytes, 336)
                .unwrap()
                .validate_child(0, &child),
            Err(DirectoryError::ChildMismatch)
        );
        encode_interior(
            2,
            1,
            &[InteriorEntry {
                upper: key(2),
                ..entry
            }],
            &mut parent_bytes,
        )
        .unwrap();
        assert_eq!(
            decode(&parent_bytes, 336)
                .unwrap()
                .validate_child(0, &child),
            Err(DirectoryError::ChildMismatch)
        );
        let mut bad_leaf = child_bytes;
        // Payload descriptors are also bounded before their containing leaf.
        put_u64(&mut bad_leaf, 48, 249);
        assert_eq!(
            decode(&bad_leaf, 256).unwrap_err(),
            DirectoryError::ForwardReference
        );
        for offset in [0, 63, 32_768, u64::MAX] {
            put_u64(&mut bad_leaf, 48, offset);
            assert!(decode(&bad_leaf, 256).is_err());
        }
    }

    #[test]
    fn directory_sibling_ranges_must_be_strictly_disjoint() {
        let a = InteriorEntry {
            lower: key(1),
            upper: key(5),
            child: page(256, 80),
        };
        let mut bytes = [0; 160];
        for b in [
            InteriorEntry {
                lower: key(5),
                upper: key(9),
                ..a
            },
            InteriorEntry {
                lower: key(2),
                upper: key(3),
                ..a
            },
            InteriorEntry {
                lower: key(9),
                upper: key(6),
                ..a
            },
        ] {
            assert_eq!(
                encode_interior(2, 2, &[a, b], &mut bytes),
                Err(DirectoryError::KeyOrder)
            );
        }
        let b = InteriorEntry {
            lower: key(6),
            upper: key(9),
            child: page(336, 80),
        };
        encode_interior(2, 2, &[a, b], &mut bytes).unwrap();
        decode(&bytes, 512).unwrap();
        put_u64(&mut bytes, 104, 5); // Second lower ordinal overlaps first upper.
        assert_eq!(decode(&bytes, 512).unwrap_err(), DirectoryError::KeyOrder);
        assert_eq!(
            encode_interior(9, 2, &[a, b], &mut bytes),
            Err(DirectoryError::Depth)
        );
        assert_eq!(
            encode_interior(2, 1, &[a, b], &mut bytes),
            Err(DirectoryError::EntryCount)
        );
    }

    #[test]
    fn directory_child_root_extents_must_follow_physical_order() {
        let left = InteriorEntry {
            lower: key(0),
            upper: key(0),
            child: page(128, 80),
        };
        let right = InteriorEntry {
            lower: key(1),
            upper: key(1),
            child: page(208, 80),
        };
        let mut bytes = [0; 160];
        encode_interior(2, 2, &[left, right], &mut bytes).unwrap();
        decode(&bytes, 512).unwrap();
        for offset in [127, 128, 207] {
            let invalid = InteriorEntry {
                child: page(offset, 80),
                ..right
            };
            let mut output = [0xa5; 160];
            assert_eq!(
                encode_interior(2, 2, &[left, invalid], &mut output),
                Err(DirectoryError::PhysicalOrder)
            );
            assert_eq!(output, [0xa5; 160]);
            let mut bad = bytes;
            put_u64(&mut bad, 128, offset);
            assert_eq!(
                decode(&bad, 512).unwrap_err(),
                DirectoryError::PhysicalOrder
            );
        }
    }

    #[test]
    fn directory_encoder_errors_do_not_mutate_output() {
        let mut output = [0xa5; 160];
        let invalid_page = LeafEntry {
            page: page(64, 0),
            ..leaf(2)
        };
        assert!(encode_leaf(&[leaf(1), invalid_page], &mut output).is_err());
        assert_eq!(output, [0xa5; 160]);
        assert_eq!(encode_leaf(&[], &mut output), Err(DirectoryError::Fanout));
        assert_eq!(
            encode_leaf(&[leaf(1)], &mut output[..79]),
            Err(DirectoryError::OutputTooShort)
        );
        let too_many = [leaf(1); MAX_FANOUT + 1];
        assert_eq!(
            encode_leaf(&too_many, &mut output),
            Err(DirectoryError::Fanout)
        );
        let valid = InteriorEntry {
            lower: key(1),
            upper: key(1),
            child: page(256, 80),
        };
        let invalid = InteriorEntry {
            lower: key(2),
            upper: key(2),
            child: page(256, 0),
        };
        assert!(encode_interior(2, 2, &[valid, invalid], &mut output).is_err());
        assert_eq!(output, [0xa5; 160]);
    }

    #[test]
    fn directory_root_empty_and_extreme_signed_bounds() {
        let empty = RootSummary {
            layout: Layout::RowId,
            legacy_base: None,
            row_count: 0,
            column_count: 0,
            group_count: 0,
            entry_count: 0,
            rows: None,
            window: None,
            directory: None,
        };
        let empty_footer = Footer {
            file_length: 256,
            root: page(64, 128),
        };
        let bytes = empty.encode().unwrap();
        assert_eq!(
            RootSummary::decode(&bytes, &empty_footer, &limits()).unwrap(),
            empty
        );
        assert!(bytes[8..].iter().all(|&b| b == 0));
        let full_range = RootSummary {
            row_count: u64::MAX,
            rows: Some(RowBounds {
                min: i64::MIN,
                max: i64::MAX,
            }),
            ..root()
        };
        assert_eq!(
            RootSummary::decode(&full_range.encode().unwrap(), &footer(), &limits()).unwrap(),
            full_range
        );
        let zero_key = RootSummary {
            row_count: 1,
            rows: Some(RowBounds { min: 0, max: 0 }),
            ..root()
        };
        assert!(zero_key.encode().is_ok());
        let one_window = RootSummary {
            window: Some(WindowBounds { lower: 1, upper: 1 }),
            ..root()
        };
        assert_eq!(
            RootSummary::decode(&one_window.encode().unwrap(), &footer(), &limits()).unwrap(),
            one_window
        );
        for invalid in [
            RootSummary {
                row_count: 0,
                ..root()
            },
            RootSummary {
                group_count: 0,
                ..root()
            },
            RootSummary {
                group_count: 3,
                ..root()
            },
            RootSummary {
                rows: None,
                ..root()
            },
            RootSummary {
                directory: None,
                ..root()
            },
            RootSummary {
                entry_count: 0,
                ..root()
            },
            RootSummary {
                rows: Some(RowBounds { min: 1, max: 0 }),
                ..root()
            },
            RootSummary {
                rows: Some(RowBounds { min: 0, max: 0 }),
                ..root()
            },
            RootSummary {
                window: Some(WindowBounds { lower: 2, upper: 1 }),
                ..root()
            },
            RootSummary {
                directory: Some(DirectoryRoot {
                    depth: 9,
                    page: page(256, 80),
                }),
                ..root()
            },
        ] {
            assert!(invalid.encode().is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn directory_root_rejects_trailing_reserved_and_absent_fields() {
        let bytes = root().encode().unwrap();
        for len in 0..128 {
            assert!(RootSummary::decode(&bytes[..len], &footer(), &limits()).is_err());
        }
        let mut longer = [0; 129];
        longer[..128].copy_from_slice(&bytes);
        assert!(RootSummary::decode(&longer, &footer(), &limits()).is_err());
        for (position, value, expected) in [
            (0, 0, DirectoryError::Magic),
            (4, 2, DirectoryError::Revision),
            (6, 0, DirectoryError::Length),
            (8, 31, DirectoryError::PresenceFlags),
            (12, 2, DirectoryError::Layout),
            (14, 1, DirectoryError::Reserved),
            (28, 1, DirectoryError::Reserved),
            (112, 1, DirectoryError::Reserved),
        ] {
            let mut bad = bytes;
            bad[position] = value;
            assert_eq!(
                RootSummary::decode(&bad, &footer(), &limits()),
                Err(expected)
            );
        }
        for bit in [HAS_ROWS, HAS_WINDOW, HAS_DIRECTORY] {
            let mut bad = bytes;
            put_u32(&mut bad, 8, (HAS_ROWS | HAS_WINDOW | HAS_DIRECTORY) & !bit);
            assert!(RootSummary::decode(&bad, &footer(), &limits()).is_err());
        }
        let mut bad = bytes;
        put_u64(&mut bad, 80, 32_768);
        assert!(RootSummary::decode(&bad, &footer(), &limits()).is_err());
        let mut child_bytes = [0; 80];
        encode_leaf(&[leaf(1)], &mut child_bytes).unwrap();
        let child = DirectoryNode::decode(
            &child_bytes,
            root().directory.unwrap().page,
            &footer(),
            &limits(),
        )
        .unwrap();
        root().validate_directory_root(&child).unwrap();
        assert_eq!(
            RootSummary {
                entry_count: 2,
                ..root()
            }
            .validate_directory_root(&child),
            Err(DirectoryError::ChildMismatch)
        );
        assert!(RootSummary {
            row_count: 0,
            ..root()
        }
        .validate_directory_root(&child)
        .is_err());
    }
}
