// Copyright 2025 Stoolap Contributors
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

//! The V5 volume file: a header, the column blocks, then sections that a
//! fixed footer locates through a section table. Every section and every
//! block carries its own CRC-32C; an open reads the footer, the table and
//! the sections, and a block is checked each time it is read.

use std::io::{self, Write};

use super::format::V5Section;
use super::writer::FileIdentity;

pub(crate) const V5_MAGIC: [u8; 4] = *b"STV5";
/// The format this writer produces
const FORMAT_VERSION: u32 = 1;
/// The newest `min_reader_version` this reader understands
const READER_VERSION: u32 = 1;
/// Magic, format version, min reader version, reserved
pub(crate) const HEADER_LEN: u64 = 16;
/// Section table offset and length, its CRC, footer version, the CRC of
/// those 24 bytes, magic
const FOOTER_LEN: u64 = 32;
const FOOTER_VERSION: u32 = 1;
/// Section entries: kind, flags, offset, stored and decoded length,
/// codec, encoding, reserved, CRC
const SECTION_ENTRY: usize = 40;
/// Block entries: offset, stored and decoded length, column, group,
/// codec, reserved, CRC
const BLOCK_ENTRY: usize = 40;
const ROOT_LEN: usize = 64;

/// A section a reader must understand to read the volume
const REQUIRED: u32 = 1;
const SECTION_ROOT: u32 = 1;
const SECTION_BLOCKS: u32 = 2;

const CODEC_RAW: u8 = 0;
const CODEC_LZ4: u8 = 1;
/// LZ4 never expands more than this, so a larger decoded length is a lie
const LZ4_MAX_RATIO: u64 = 255;

const CHECKSUM_CRC32C: u8 = 1;
const LITTLE_ENDIAN: u8 = 1;
/// Blocks laid out group by group; informational, offsets are explicit
const PLACEMENT_GROUP_MAJOR: u8 = 0;

/// The checksum of every V5 section and block
#[inline]
pub(crate) fn checksum(bytes: &[u8]) -> u32 {
    crc32c::crc32c(bytes)
}

pub(crate) fn header() -> [u8; HEADER_LEN as usize] {
    let mut h = [0u8; HEADER_LEN as usize];
    h[0..4].copy_from_slice(&V5_MAGIC);
    h[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    h[8..12].copy_from_slice(&READER_VERSION.to_le_bytes());
    h
}

/// One block as written: where, how long stored and decoded, its codec
/// and its CRC
#[derive(Clone, Copy)]
pub(crate) struct BlockEntry {
    offset: u64,
    stored: u64,
    decoded: u64,
    col: u32,
    group: u32,
    codec: u8,
    crc: u32,
}

impl BlockEntry {
    /// The entry of `stored`, kept raw when it is as long as the decoded
    /// block, LZ4 otherwise, as both writers store blocks
    pub(crate) fn new(
        offset: u64,
        stored: &[u8],
        decoded: usize,
        col: usize,
        group: usize,
    ) -> io::Result<Self> {
        let narrow = |v: usize| {
            u32::try_from(v).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "V5 block index exceeds u32")
            })
        };
        Ok(Self {
            offset,
            stored: stored.len() as u64,
            decoded: decoded as u64,
            col: narrow(col)?,
            group: narrow(group)?,
            codec: if stored.len() == decoded {
                CODEC_RAW
            } else {
                CODEC_LZ4
            },
            crc: checksum(stored),
        })
    }

    pub(crate) fn crc(&self) -> u32 {
        self.crc
    }
}

/// The volume's shape and identity the root records
pub(crate) struct Root {
    pub cols: u32,
    pub groups: u32,
    pub group_rows: u32,
    pub rows: u64,
    pub volume_id: u64,
    /// Random per file, so a file can be told from another with the same
    /// id and length
    pub nonce: u64,
}

/// Writes the sections, the section table and the footer after the
/// blocks, which end at `pos`; returns the identity of the file written
pub(crate) fn write_tail<W: Write>(
    out: &mut W,
    mut pos: u64,
    root: &Root,
    meta: Vec<V5Section>,
    blocks: &[BlockEntry],
) -> io::Result<FileIdentity> {
    let mut root_bytes = Vec::with_capacity(ROOT_LEN);
    root_bytes.extend_from_slice(&(ROOT_LEN as u32).to_le_bytes());
    root_bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    root_bytes.extend_from_slice(&READER_VERSION.to_le_bytes());
    root_bytes.extend_from_slice(&root.cols.to_le_bytes());
    root_bytes.extend_from_slice(&root.groups.to_le_bytes());
    root_bytes.extend_from_slice(&root.group_rows.to_le_bytes());
    root_bytes.extend_from_slice(&root.rows.to_le_bytes());
    root_bytes.extend_from_slice(&[PLACEMENT_GROUP_MAJOR, CHECKSUM_CRC32C, LITTLE_ENDIAN, 0]);
    root_bytes.extend_from_slice(&(BLOCK_ENTRY as u32).to_le_bytes());
    root_bytes.extend_from_slice(&root.volume_id.to_le_bytes());
    root_bytes.extend_from_slice(&root.nonce.to_le_bytes());
    root_bytes.resize(ROOT_LEN, 0);

    let mut table = Vec::with_capacity(blocks.len() * BLOCK_ENTRY);
    for b in blocks {
        table.extend_from_slice(&b.offset.to_le_bytes());
        table.extend_from_slice(&b.stored.to_le_bytes());
        table.extend_from_slice(&b.decoded.to_le_bytes());
        table.extend_from_slice(&b.col.to_le_bytes());
        table.extend_from_slice(&b.group.to_le_bytes());
        table.extend_from_slice(&[b.codec, 0, 0, 0]);
        table.extend_from_slice(&b.crc.to_le_bytes());
    }

    let mut all = Vec::with_capacity(meta.len() + 2);
    all.push(V5Section {
        kind: SECTION_ROOT,
        required: true,
        encoding: 1,
        compress: false,
        bytes: root_bytes,
    });
    all.push(V5Section {
        kind: SECTION_BLOCKS,
        required: true,
        encoding: 1,
        compress: false,
        bytes: table,
    });
    all.extend(meta);

    let mut sections = Vec::with_capacity(8 + all.len() * SECTION_ENTRY);
    sections.extend_from_slice(&(all.len() as u32).to_le_bytes());
    sections.extend_from_slice(&(SECTION_ENTRY as u32).to_le_bytes());
    for s in &all {
        let packed;
        let (stored, codec): (&[u8], u8) = if s.compress {
            packed = lz4_flex::block::compress(&s.bytes);
            if packed.len() < s.bytes.len() {
                (&packed, CODEC_LZ4)
            } else {
                (&s.bytes, CODEC_RAW)
            }
        } else {
            (&s.bytes, CODEC_RAW)
        };
        out.write_all(stored)?;
        sections.extend_from_slice(&s.kind.to_le_bytes());
        let flags = if s.required { REQUIRED } else { 0 };
        sections.extend_from_slice(&flags.to_le_bytes());
        sections.extend_from_slice(&pos.to_le_bytes());
        sections.extend_from_slice(&(stored.len() as u64).to_le_bytes());
        sections.extend_from_slice(&(s.bytes.len() as u64).to_le_bytes());
        sections.extend_from_slice(&[codec, s.encoding, 0, 0]);
        sections.extend_from_slice(&checksum(stored).to_le_bytes());
        pos += stored.len() as u64;
    }
    out.write_all(&sections)?;

    let table_crc = checksum(&sections);
    let mut footer = [0u8; FOOTER_LEN as usize];
    footer[0..8].copy_from_slice(&pos.to_le_bytes());
    footer[8..16].copy_from_slice(&(sections.len() as u64).to_le_bytes());
    footer[16..20].copy_from_slice(&table_crc.to_le_bytes());
    footer[20..24].copy_from_slice(&FOOTER_VERSION.to_le_bytes());
    let footer_crc = checksum(&footer[0..24]);
    footer[24..28].copy_from_slice(&footer_crc.to_le_bytes());
    footer[28..32].copy_from_slice(&V5_MAGIC);
    out.write_all(&footer)?;
    Ok(FileIdentity {
        len: pos + sections.len() as u64 + FOOTER_LEN,
        nonce: root.nonce,
        table_crc,
    })
}

/// What an open reads of a V5 volume besides the blocks
pub(crate) struct Opened {
    pub root: Root,
    /// The metadata sections this reader knows: (kind, encoding, bytes)
    pub meta: Vec<(u32, u8, Vec<u8>)>,
    /// The sections left in the file: (kind, encoding, offset, length),
    /// each stored raw
    pub deferred: Vec<(u32, u8, u64, u64)>,
    /// The block table, its entry size and where the sections begin
    blocks: Vec<u8>,
    block_entry: usize,
    sections_start: u64,
    pub identity: FileIdentity,
}

/// Every block's place, by column, by group
pub(crate) struct Places {
    pub offsets: Vec<Vec<u64>>,
    pub stored: Vec<Vec<usize>>,
    pub decoded: Vec<Vec<usize>>,
    pub crcs: Vec<Vec<u32>>,
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("V5: {}", msg.into()))
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    let mut v = [0u8; 8];
    v.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(v)
}

/// A length read from the file as an in-memory size, refused past `limit`
/// before anything is allocated for it
fn bounded(len: u64, limit: u64, what: &str) -> io::Result<usize> {
    if len > limit {
        return Err(invalid(format!("{what} length exceeds its bound")));
    }
    usize::try_from(len).map_err(|_| invalid(format!("{what} length exceeds the address space")))
}

/// Reads the footer, the section table and the sections `known` names
/// (besides the root and the block table) of the file `read_at` reads by
/// position, `len` bytes long; the `deferred` ones' places only, their
/// readers checking what they read
pub(crate) fn open(
    len: u64,
    known: &[u32],
    deferred: &[u32],
    read_at: &dyn Fn(&mut [u8], u64) -> io::Result<()>,
) -> io::Result<Opened> {
    if len < HEADER_LEN + FOOTER_LEN {
        return Err(invalid("file too small"));
    }
    let mut header = [0u8; HEADER_LEN as usize];
    read_at(&mut header, 0)?;
    if header[0..4] != V5_MAGIC {
        return Err(invalid("bad magic"));
    }
    if u32_at(&header, 8) > READER_VERSION {
        return Err(invalid("written for a newer reader"));
    }
    let mut footer = [0u8; FOOTER_LEN as usize];
    read_at(&mut footer, len - FOOTER_LEN)?;
    if footer[28..32] != V5_MAGIC
        || checksum(&footer[0..24]) != u32_at(&footer, 24)
        || u32_at(&footer, 20) != FOOTER_VERSION
    {
        return Err(invalid("bad footer"));
    }
    let tail_end = len - FOOTER_LEN;
    let (table_at, table_len) = (u64_at(&footer, 0), u64_at(&footer, 8));
    if table_at < HEADER_LEN || table_at.checked_add(table_len).is_none_or(|e| e > tail_end) {
        return Err(invalid("section table outside the file"));
    }
    let mut table = vec![0u8; bounded(table_len, tail_end, "section table")?];
    read_at(&mut table, table_at)?;
    if checksum(&table) != u32_at(&footer, 16) || table.len() < 8 {
        return Err(invalid("section table CRC mismatch"));
    }
    let (count, entry) = (u32_at(&table, 0) as usize, u32_at(&table, 4) as usize);
    if entry < SECTION_ENTRY
        || count
            .checked_mul(entry)
            .and_then(|n| n.checked_add(8))
            .is_none_or(|n| n != table.len())
    {
        return Err(invalid("section table size"));
    }
    let mut root = None;
    let mut blocks = None;
    let mut meta: Vec<(u32, u8, Vec<u8>)> = Vec::with_capacity(count);
    let mut left = Vec::new();
    let mut seen = Vec::with_capacity(count);
    let mut sections_start = table_at;
    for i in 0..count {
        let e = &table[8 + i * entry..8 + (i + 1) * entry];
        let (kind, flags, at, stored, decoded) = (
            u32_at(e, 0),
            u32_at(e, 4),
            u64_at(e, 8),
            u64_at(e, 16),
            u64_at(e, 24),
        );
        let (codec, encoding, crc) = (e[32], e[33], u32_at(e, 36));
        if seen.contains(&kind) {
            return Err(invalid(format!("section {kind} appears twice")));
        }
        seen.push(kind);
        if at < HEADER_LEN || at.checked_add(stored).is_none_or(|end| end > table_at) {
            return Err(invalid("section outside the file"));
        }
        sections_start = sections_start.min(at);
        if deferred.contains(&kind) {
            if codec != CODEC_RAW || stored != decoded {
                return Err(invalid(format!("section {kind} is not stored raw")));
            }
            left.push((kind, encoding, at, stored));
            continue;
        }
        let wanted = kind == SECTION_ROOT || kind == SECTION_BLOCKS || known.contains(&kind);
        if !wanted {
            if flags & REQUIRED != 0 {
                return Err(invalid(format!("unknown required section {kind}")));
            }
            continue;
        }
        let mut bytes = vec![0u8; bounded(stored, table_at, "section")?];
        read_at(&mut bytes, at)?;
        if checksum(&bytes) != crc {
            return Err(invalid(format!("section {kind} CRC mismatch")));
        }
        let bytes = match codec {
            CODEC_RAW if stored == decoded => bytes,
            CODEC_LZ4 => {
                let limit = stored.saturating_mul(LZ4_MAX_RATIO).saturating_add(16);
                let decoded = bounded(decoded, limit, "decoded section")?;
                let out = lz4_flex::block::decompress(&bytes, decoded)
                    .map_err(|e| invalid(format!("section {kind}: {e}")))?;
                if out.len() != decoded {
                    return Err(invalid(format!("section {kind} decoded length")));
                }
                out
            }
            _ => return Err(invalid(format!("section {kind} codec {codec}"))),
        };
        match kind {
            SECTION_ROOT => root = Some((encoding, bytes)),
            SECTION_BLOCKS => blocks = Some((encoding, bytes)),
            _ => meta.push((kind, encoding, bytes)),
        }
    }
    let ((root_encoding, root), (block_encoding, blocks)) = match (root, blocks) {
        (Some(r), Some(b)) => (r, b),
        _ => return Err(invalid("a required section is missing")),
    };
    if root_encoding != 1 {
        return Err(invalid("root encoding"));
    }
    if root.len() < ROOT_LEN || u32_at(&root, 0) as usize != root.len() {
        return Err(invalid("root length"));
    }
    if u32_at(&root, 8) > READER_VERSION {
        return Err(invalid("written for a newer reader"));
    }
    if root[33] != CHECKSUM_CRC32C || root[34] != LITTLE_ENDIAN {
        return Err(invalid(
            "root names a checksum or byte order this reader lacks",
        ));
    }
    let block_entry = u32_at(&root, 36) as usize;
    let root = Root {
        cols: u32_at(&root, 12),
        groups: u32_at(&root, 16),
        group_rows: u32_at(&root, 20),
        rows: u64_at(&root, 24),
        volume_id: u64_at(&root, 40),
        nonce: u64_at(&root, 48),
    };
    if block_encoding != 1 || block_entry < BLOCK_ENTRY {
        return Err(invalid("block table encoding"));
    }
    // One entry per column per group, checked without sizing anything
    if (root.cols as usize)
        .checked_mul(root.groups as usize)
        .and_then(|n| n.checked_mul(block_entry))
        .is_none_or(|n| n != blocks.len())
    {
        return Err(invalid("block table size"));
    }
    let identity = FileIdentity {
        len,
        nonce: root.nonce,
        table_crc: u32_at(&footer, 16),
    };
    Ok(Opened {
        root,
        meta,
        deferred: left,
        blocks,
        block_entry,
        sections_start,
        identity,
    })
}

impl Opened {
    /// The blocks' places, for the column and group counts the metadata
    /// gives; refused unless the root and the block table agree with them
    pub(crate) fn places(&self, cols: usize, groups: usize) -> io::Result<Places> {
        if cols != self.root.cols as usize || groups != self.root.groups as usize {
            return Err(invalid("block table size"));
        }
        let mut offsets = vec![vec![u64::MAX; groups]; cols];
        let mut stored = vec![vec![0usize; groups]; cols];
        let mut decoded = vec![vec![0usize; groups]; cols];
        let mut crcs = vec![vec![0u32; groups]; cols];
        let sections_start = self.sections_start;
        for e in self.blocks.chunks_exact(self.block_entry) {
            let (at, len, dec) = (u64_at(e, 0), u64_at(e, 8), u64_at(e, 16));
            let (col, group) = (u32_at(e, 24) as usize, u32_at(e, 28) as usize);
            if col >= cols || group >= groups || offsets[col][group] != u64::MAX {
                return Err(invalid("block table names a block twice or out of range"));
            }
            if at < HEADER_LEN || at.checked_add(len).is_none_or(|end| end > sections_start) {
                return Err(invalid("block outside the block area"));
            }
            // The decoders tell a raw block by its length: the codec must agree
            let dec_limit = match e[32] {
                CODEC_RAW if len == dec => len,
                CODEC_LZ4 if len != dec => len.saturating_mul(LZ4_MAX_RATIO).saturating_add(16),
                _ => return Err(invalid("block codec")),
            };
            offsets[col][group] = at;
            stored[col][group] = bounded(len, sections_start, "block")?;
            decoded[col][group] = bounded(dec, dec_limit, "decoded block")?;
            crcs[col][group] = u32_at(e, 36);
        }
        Ok(Places {
            offsets,
            stored,
            decoded,
            crcs,
        })
    }
}

/// A random nonce for a new volume file
pub(crate) fn new_nonce() -> u64 {
    rand::random::<u64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: &[u8] = b"abcd";

    fn section(kind: u32, required: bool, compress: bool, bytes: Vec<u8>) -> V5Section {
        V5Section {
            kind,
            required,
            encoding: 1,
            compress,
            bytes,
        }
    }

    /// One raw block of one column in one group, then `meta` and the tail
    fn file(meta: Vec<V5Section>) -> Vec<u8> {
        let mut out = header().to_vec();
        let entry = BlockEntry::new(HEADER_LEN, BLOCK, BLOCK.len(), 0, 0).unwrap();
        out.extend_from_slice(BLOCK);
        let root = Root {
            cols: 1,
            groups: 1,
            group_rows: 4,
            rows: 4,
            volume_id: 3,
            nonce: new_nonce(),
        };
        let pos = out.len() as u64;
        write_tail(&mut out, pos, &root, meta, &[entry]).unwrap();
        out
    }

    fn known_meta() -> Vec<V5Section> {
        vec![
            section(5, true, false, vec![7; 24]),
            section(6, true, true, vec![b'x'; 300]),
        ]
    }

    fn open_bytes(bytes: &[u8], known: &[u32]) -> io::Result<Opened> {
        let read_at = |buf: &mut [u8], at: u64| {
            let at = at as usize;
            buf.copy_from_slice(&bytes[at..at + buf.len()]);
            Ok(())
        };
        open(bytes.len() as u64, known, &[], &read_at)
    }

    #[test]
    fn an_intact_file_opens_with_its_root_blocks_and_sections() {
        let bytes = file(known_meta());
        let opened = open_bytes(&bytes, &[5, 6]).unwrap();
        let root = &opened.root;
        assert_eq!(
            (
                root.cols,
                root.groups,
                root.group_rows,
                root.rows,
                root.volume_id
            ),
            (1, 1, 4, 4, 3)
        );
        let places = opened.places(1, 1).unwrap();
        assert_eq!(places.offsets, vec![vec![HEADER_LEN]]);
        assert_eq!(places.stored, vec![vec![BLOCK.len()]]);
        assert_eq!(places.decoded, vec![vec![BLOCK.len()]]);
        assert_eq!(places.crcs, vec![vec![checksum(BLOCK)]]);
        assert_eq!(
            opened.meta,
            vec![(5, 1, vec![7; 24]), (6, 1, vec![b'x'; 300])]
        );
    }

    /// Edits the section table and seals the footer over it again
    fn edit_table(bytes: &mut [u8], edit: impl FnOnce(&mut [u8])) {
        let footer = bytes.len() - FOOTER_LEN as usize;
        let at = u64_at(bytes, footer) as usize;
        let len = u64_at(bytes, footer + 8) as usize;
        edit(&mut bytes[at..at + len]);
        let crc = checksum(&bytes[at..at + len]);
        bytes[footer + 16..footer + 20].copy_from_slice(&crc.to_le_bytes());
        let crc = checksum(&bytes[footer..footer + 24]);
        bytes[footer + 24..footer + 28].copy_from_slice(&crc.to_le_bytes());
    }

    #[test]
    fn a_root_in_an_unknown_encoding_is_refused() {
        let mut bytes = file(known_meta());
        // The root is the table's first entry; byte 33 is its encoding
        edit_table(&mut bytes, |table| table[8 + 33] = 2);
        let error = open_bytes(&bytes, &[5, 6]).err().unwrap();
        assert!(error.to_string().contains("root encoding"), "{error}");
    }

    #[test]
    fn block_places_are_sized_only_by_counts_the_metadata_confirms() {
        let mut out = header().to_vec();
        let root = Root {
            cols: u32::MAX,
            groups: 0,
            group_rows: 4,
            rows: 0,
            volume_id: 3,
            nonce: new_nonce(),
        };
        let pos = out.len() as u64;
        write_tail(&mut out, pos, &root, known_meta(), &[]).unwrap();
        let opened = open_bytes(&out, &[5, 6]).unwrap();
        let error = opened.places(1, 0).err().unwrap();
        assert!(error.to_string().contains("block table size"), "{error}");
    }

    #[test]
    fn a_root_whose_groups_have_no_blocks_is_refused_at_open() {
        let mut out = header().to_vec();
        let root = Root {
            cols: 1,
            groups: 16_384,
            group_rows: 65_536,
            rows: 1 << 30,
            volume_id: 3,
            nonce: new_nonce(),
        };
        let pos = out.len() as u64;
        write_tail(&mut out, pos, &root, known_meta(), &[]).unwrap();
        let error = open_bytes(&out, &[5, 6]).err().unwrap();
        assert!(error.to_string().contains("block table size"), "{error}");
    }

    #[test]
    fn every_flipped_byte_after_the_blocks_is_refused() {
        let bytes = file(known_meta());
        let tail = HEADER_LEN as usize + BLOCK.len();
        for at in tail..bytes.len() {
            let mut damaged = bytes.clone();
            damaged[at] ^= 0x01;
            assert!(
                open_bytes(&damaged, &[5, 6]).is_err(),
                "a flipped byte at {at} of {} was accepted",
                bytes.len()
            );
        }
    }

    #[test]
    fn every_truncation_is_refused() {
        let bytes = file(known_meta());
        for len in 0..bytes.len() {
            assert!(
                open_bytes(&bytes[..len], &[5, 6]).is_err(),
                "the first {len} bytes were accepted"
            );
        }
    }

    #[test]
    fn an_unknown_required_section_is_refused_and_an_unknown_optional_one_skipped() {
        let mut meta = known_meta();
        meta.push(section(99, false, false, vec![1; 8]));
        let opened = open_bytes(&file(meta), &[5, 6]).unwrap();
        assert!(opened.meta.iter().all(|(kind, _, _)| *kind != 99));

        let mut meta = known_meta();
        meta.push(section(99, true, false, vec![1; 8]));
        let error = open_bytes(&file(meta), &[5, 6]).err().unwrap();
        assert!(
            error.to_string().contains("unknown required section 99"),
            "{error}"
        );
    }

    #[test]
    fn a_section_listed_twice_is_refused() {
        let mut meta = known_meta();
        meta.push(section(5, true, false, vec![7; 24]));
        let error = open_bytes(&file(meta), &[5, 6]).err().unwrap();
        assert!(error.to_string().contains("appears twice"), "{error}");
    }

    #[test]
    fn a_file_for_a_newer_reader_is_refused() {
        let mut bytes = file(known_meta());
        bytes[8..12].copy_from_slice(&(READER_VERSION + 1).to_le_bytes());
        let error = open_bytes(&bytes, &[5, 6]).err().unwrap();
        assert!(error.to_string().contains("newer reader"), "{error}");
    }
}
