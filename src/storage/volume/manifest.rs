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

//! Table manifest: the source of truth for a table's segment state.
//!
//! The manifest tracks which immutable segments (frozen volumes) exist for a
//! table, along with a tombstone set of cold row_ids that have been deleted
//! or superseded by hot buffer versions.
//!
//! # Persistence
//!
//! The manifest is written atomically (tmp + rename) to a single file
//! per table. On recovery, the manifest is loaded first, then segments
//! are loaded from the paths recorded in the manifest.

use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::common::SmartString;
use crate::core::{Result, Value};

use super::writer::FrozenVolume;

/// A cold segment: immutable volume + pre-computed column mapping.
/// The mapping is computed once at registration (seal/compaction/load) and
/// recomputed on ALTER TABLE. No per-scan computation, no lock contention.
#[derive(Clone)]
pub struct ColdSegment {
    pub volume: Arc<FrozenVolume>,
    pub mapping: super::writer::ColumnMapping,
    /// Schema version when this volume was created. Used with dropped_columns
    /// to correctly mask stale data only from volumes older than a column drop.
    pub schema_version: u64,
    /// Per-row visibility bitmap: bit i is set when row i is the authoritative
    /// (newest) version across all overlapping volumes. None when this is the
    /// only segment (all rows visible) or when there is no overlap.
    /// Arc so ColdSegment::clone() is O(1) — scanners share the same bitmap.
    pub visible: Option<Arc<Vec<u64>>>,
}

/// Versioned tombstone map: row_id -> commit_seq at which the tombstone was published.
pub type TombstoneMap = Arc<FxHashMap<i64, u64>>;

/// Ordered cold-segment list paired with their segment ids.
pub type SegmentList = Vec<(u64, ColdSegment)>;

/// Shared snapshot of the segment list (Arc-wrapped for cheap cloning).
pub type SharedSegmentList = Arc<SegmentList>;

impl ColdSegment {
    /// Check whether row at position `idx` in this volume is the authoritative
    /// (newest) copy across all overlapping volumes.
    #[inline]
    pub fn is_visible(&self, idx: usize) -> bool {
        match &self.visible {
            None => true,
            Some(bits) => (bits[idx >> 6] >> (idx & 63)) & 1 == 1,
        }
    }
}

/// Atomic snapshot of cold segment state for batch constraint checking.
/// Captures manifest seg_ids + segments Arc + tombstones Arc once,
/// eliminating 3 lock reads per row in batch INSERT/upsert.
pub struct ColdSnapshot {
    pub seg_ids: smallvec::SmallVec<[u64; 4]>,
    pub segs: Arc<FxHashMap<u64, ColdSegment>>,
    pub ts: Arc<FxHashMap<i64, u64>>,
}

// Manifest file magic: "STMF" (SToolap ManiFest)
const MANIFEST_MAGIC: [u8; 4] = *b"STMF";
/// Current manifest format version.
///
/// V7 (SWMR v2) is the single SWMR format. Relative to V6 (the
/// last released format), V7:
///   - adds per-segment `visible_at_lsn` so capped read-only
///     attaches drop post-cap segments,
///   - widens tombstones from 16-byte `(row_id, commit_seq)`
///     entries to 24-byte `(row_id, commit_seq, visible_at_lsn)`
///     triples so capped attaches drop post-cap tombstones,
///   - adds a per-table `schema_version` (engine's `schema_epoch`
///     at manifest-write time) so a no-shm reader can detect
///     ALTER TABLE ADD/MODIFY/DROP/RENAME COLUMN that did not
///     produce a new segment with a bumped per-segment
///     `schema_version`.
///
/// V6 manifests deserialize: per-segment `visible_at_lsn`
/// synthesizes from `creation_lsn`, tombstones synthesize
/// `visible_at_lsn = 0` ("always visible"), and table-level
/// `schema_version` synthesizes as `0` ("no DDL recorded yet").
const MANIFEST_VERSION: u32 = 7;
/// V6 (legacy v0.4.0) manifest version. Deserialize-only; the
/// next write upgrades in place to V7.
const MANIFEST_VERSION_V6: u32 = 6;

/// Metadata for a single immutable segment (frozen volume).
#[derive(Debug, Clone)]
pub struct SegmentMeta {
    /// Unique segment identifier (monotonically increasing).
    pub segment_id: u64,
    /// Path to the segment file on disk (relative to volume dir).
    pub file_path: PathBuf,
    /// Number of rows in this segment.
    pub row_count: usize,
    /// Minimum row_id in this segment.
    pub min_row_id: i64,
    /// Maximum row_id in this segment.
    pub max_row_id: i64,
    /// WAL LSN at which this segment was created (for recovery).
    pub creation_lsn: u64,
    /// Commit sequence at which this volume was sealed. Used to gate compaction:
    /// volumes with seal_seq > min_snapshot_begin_seq are not compacted.
    /// Old volumes (pre-tracking) have seal_seq=0, treated as "always safe".
    pub seal_seq: u64,
    /// Schema version when this segment was created. Used with dropped_columns
    /// to correctly mask stale data only from volumes older than a column drop.
    pub schema_version: u64,
    /// SWMR v2 Phase F: WAL LSN at which this segment first became
    /// visible to readers (the LSN of the manifest write that
    /// introduced it). A snapshot pinned at `visible_commit_lsn = P`
    /// must NOT see segments with `visible_at_lsn > P`. For segments
    /// loaded from a V6 manifest this is synthesized from
    /// `creation_lsn` on read (the closest-available approximation).
    /// New segments set this to the current WAL LSN at registration.
    pub visible_at_lsn: u64,
}

impl SegmentMeta {
    /// SWMR v2 Phase F: is this segment visible to a reader pinned at
    /// `pinned_lsn`? `pinned_lsn = 0` means "no LSN pin" — see all
    /// segments (used by writer-side scans). Otherwise the segment is
    /// visible iff `visible_at_lsn <= pinned_lsn`.
    #[inline]
    pub fn is_visible_to_lsn(&self, pinned_lsn: u64) -> bool {
        pinned_lsn == 0 || self.visible_at_lsn <= pinned_lsn
    }
}

/// The manifest for a single table: tracks all live segments and tombstones.
///
/// This is the source of truth for what segments exist and which cold
/// row_ids have been deleted or superseded.
#[derive(Debug, Clone)]
pub struct TableManifest {
    /// Table name.
    pub table_name: SmartString,
    /// Live segments (ordered by segment_id, oldest first).
    pub segments: Vec<SegmentMeta>,
    /// Next segment ID to assign.
    pub next_segment_id: u64,
    /// WAL LSN of the last checkpoint that included this manifest.
    pub checkpoint_lsn: u64,
    /// Tombstone entries: (row_id, commit_seq, visible_at_lsn) triples
    /// for cold rows that have been deleted or superseded.
    ///
    /// - `commit_seq`: when the tombstone was created in the writer's
    ///   commit-sequence space. Enables snapshot isolation — a
    ///   snapshot transaction at `begin_seq = N` only sees tombstones
    ///   with `commit_seq <= N`.
    /// - `visible_at_lsn` (V7): the WAL LSN at which this tombstone
    ///   first became cross-process visible. Capped read-only attaches
    ///   hide tombstones with `visible_at_lsn > cap` so they can't
    ///   suppress a cold row that was still visible at the reader's
    ///   `attach_visible_commit_lsn`. V6-loaded tombstones get
    ///   `visible_at_lsn = 0` (treated as "always visible").
    ///
    /// Cleared after compaction processes them.
    pub tombstones: Vec<(i64, u64, u64)>,
    /// Column rename history: (old_name, new_name) pairs.
    /// Applied as aliases to cold volumes on load so pre-rename data
    /// is visible through the new schema column name.
    pub column_renames: Vec<(SmartString, SmartString)>,
    /// Columns that have been dropped (and possibly re-added with same name).
    /// Each entry is (column_name, schema_version_at_drop). Old volumes sealed
    /// before the drop (schema_version <= drop_version) have stale data masked.
    /// Cleared during compaction (new volumes don't have stale data).
    pub dropped_columns: Vec<(SmartString, u64)>,
    /// Table-level schema epoch (V7). Set by the writer to its
    /// engine's `schema_epoch` whenever a DDL operation modifies
    /// this table — including ADD/MODIFY COLUMN that do not
    /// produce a new segment. The reader's drift checks
    /// (`peek_schema_drift`, `reload_from_disk`) compare this
    /// against the reader's known `schema_epoch`; a higher value
    /// means the writer has done DDL the reader hasn't applied.
    /// V6 manifests deserialize with `0` (treated as "no
    /// recorded DDL", same as a fresh V7 manifest); the next
    /// manifest write upgrades the file to V7 and stamps the
    /// current epoch.
    pub schema_version: u64,
}

impl TableManifest {
    /// Create an empty manifest for a table.
    pub fn new(table_name: &str) -> Self {
        Self {
            table_name: SmartString::from(table_name),
            segments: Vec::new(),
            next_segment_id: 1,
            checkpoint_lsn: 0,
            tombstones: Vec::new(),
            column_renames: Vec::new(),
            dropped_columns: Vec::new(),
            schema_version: 0,
        }
    }

    /// Allocate a new segment ID.
    pub fn allocate_segment_id(&mut self) -> u64 {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        id
    }

    /// Add a segment to the manifest.
    pub fn add_segment(&mut self, meta: SegmentMeta) {
        self.segments.push(meta);
    }

    /// Remove segments by ID (after compaction).
    pub fn remove_segments(&mut self, ids: &[u64]) {
        let id_set: FxHashSet<u64> = ids.iter().copied().collect();
        self.segments.retain(|s| !id_set.contains(&s.segment_id));
    }

    /// Find which segment contains a given row_id.
    ///
    /// Uses min/max row_id metadata for fast rejection.
    /// Returns (segment_index, segment_meta) if found.
    pub fn find_segment_for_row_id(&self, row_id: i64) -> Option<(usize, &SegmentMeta)> {
        for (i, seg) in self.segments.iter().enumerate() {
            if row_id >= seg.min_row_id && row_id <= seg.max_row_id {
                return Some((i, seg));
            }
        }
        None
    }

    /// Serialize the manifest to bytes (V6 format).
    pub fn serialize(&self) -> io::Result<Vec<u8>> {
        let mut buf = Vec::with_capacity(256);

        // Header
        buf.write_all(&MANIFEST_MAGIC)?;
        buf.write_all(&MANIFEST_VERSION.to_le_bytes())?;
        buf.write_all(&self.next_segment_id.to_le_bytes())?;
        buf.write_all(&self.checkpoint_lsn.to_le_bytes())?;
        // V7: table-level schema epoch — see field doc on
        // `TableManifest::schema_version` for why it's needed
        // beyond the per-segment `schema_version`.
        buf.write_all(&self.schema_version.to_le_bytes())?;

        // Table name
        let name_bytes = self.table_name.as_bytes();
        buf.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        buf.write_all(name_bytes)?;

        // Segments. V7 layout per segment (fixed fields = 64 bytes):
        //   segment_id: u64
        //   row_count: u64
        //   min_row_id: i64
        //   max_row_id: i64
        //   creation_lsn: u64
        //   seal_seq: u64
        //   schema_version: u64
        //   visible_at_lsn: u64    (V7; V6 lacks this field)
        // followed by variable-length file_path (u32 len + utf-8 bytes).
        buf.write_all(&(self.segments.len() as u32).to_le_bytes())?;
        for seg in &self.segments {
            buf.write_all(&seg.segment_id.to_le_bytes())?;
            buf.write_all(&(seg.row_count as u64).to_le_bytes())?;
            buf.write_all(&seg.min_row_id.to_le_bytes())?;
            buf.write_all(&seg.max_row_id.to_le_bytes())?;
            buf.write_all(&seg.creation_lsn.to_le_bytes())?;
            buf.write_all(&seg.seal_seq.to_le_bytes())?;
            buf.write_all(&seg.schema_version.to_le_bytes())?;
            buf.write_all(&seg.visible_at_lsn.to_le_bytes())?;

            // File path as UTF-8 string
            let path_str = seg.file_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            buf.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
            buf.write_all(path_bytes)?;
        }

        // Tombstones (V7): (row_id, commit_seq, visible_at_lsn) triples.
        // Skip ephemeral entries with `visible_at_lsn == u64::MAX` —
        // those are failed-marker tombstones (record_commit IO failure
        // after partial commit) that must stay in-memory-only. Persisting
        // them would let a future process recovery resurrect a
        // markerless commit's deletes after the WAL discarded the txn.
        let persistable_tombstones: Vec<&(i64, u64, u64)> = self
            .tombstones
            .iter()
            .filter(|(_, _, vis)| *vis != u64::MAX)
            .collect();
        buf.write_all(&(persistable_tombstones.len() as u64).to_le_bytes())?;
        for &&(row_id, commit_seq, visible_at_lsn) in &persistable_tombstones {
            buf.write_all(&row_id.to_le_bytes())?;
            buf.write_all(&commit_seq.to_le_bytes())?;
            buf.write_all(&visible_at_lsn.to_le_bytes())?;
        }

        // Column renames: (old_name, new_name) pairs
        buf.write_all(&(self.column_renames.len() as u32).to_le_bytes())?;
        for (old_name, new_name) in &self.column_renames {
            let ob = old_name.as_bytes();
            buf.write_all(&(ob.len() as u16).to_le_bytes())?;
            buf.write_all(ob)?;
            let nb = new_name.as_bytes();
            buf.write_all(&(nb.len() as u16).to_le_bytes())?;
            buf.write_all(nb)?;
        }

        // Dropped columns: (name, schema_version) pairs
        buf.write_all(&(self.dropped_columns.len() as u32).to_le_bytes())?;
        for (name, version) in &self.dropped_columns {
            let nb = name.as_bytes();
            buf.write_all(&(nb.len() as u16).to_le_bytes())?;
            buf.write_all(nb)?;
            buf.write_all(&version.to_le_bytes())?;
        }

        // Trailing CRC32 over the entire payload
        let crc = crc32fast::hash(&buf);
        buf.write_all(&crc.to_le_bytes())?;

        Ok(buf)
    }

    /// Deserialize a manifest from bytes. V6 and V7 are both
    /// supported. V6 manifests synthesize per-segment `visible_at_lsn`
    /// from `creation_lsn` (the closest-available approximation; no
    /// historical SWMR-visible-LSN exists for pre-V7 segments). The
    /// next manifest write upgrades the file in place to V7.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.len() < 28 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest too small",
            ));
        }
        if data[0..4] != MANIFEST_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid manifest magic",
            ));
        }
        let mut pos = 4;

        let version = read_u32(data, &mut pos)?;
        if version != MANIFEST_VERSION && version != MANIFEST_VERSION_V6 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported manifest version {} (this build supports V{} and V{})",
                    version, MANIFEST_VERSION_V6, MANIFEST_VERSION
                ),
            ));
        }
        // V7 carries per-segment visible_at_lsn, 24-byte
        // tombstone triples, and a per-table schema_version.
        // V6 (legacy v0.4.0) has none of these — they
        // synthesize on read.
        let is_v7 = version == MANIFEST_VERSION;
        let tombstone_has_visible_lsn = is_v7;
        let has_table_schema_version = is_v7;

        // Verify trailing CRC32 before parsing the rest.
        let payload = &data[..data.len() - 4];
        let stored_crc = u32::from_le_bytes(
            data[data.len() - 4..]
                .try_into()
                .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "bad CRC bytes"))?,
        );
        let computed_crc = crc32fast::hash(payload);
        if stored_crc != computed_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "manifest CRC mismatch: stored={:#010x}, computed={:#010x}",
                    stored_crc, computed_crc
                ),
            ));
        }

        let next_segment_id = read_u64(data, &mut pos)?;
        let checkpoint_lsn = read_u64(data, &mut pos)?;
        // V7: table-level schema epoch sits between checkpoint_lsn
        // and the table-name length. V6 defaults to 0 (treated as
        // "no recorded DDL", same as a fresh V7 write).
        let table_schema_version = if has_table_schema_version {
            read_u64(data, &mut pos)?
        } else {
            0
        };

        // Table name
        let name_len = read_u32(data, &mut pos)? as usize;
        if pos + name_len > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "manifest truncated at table name",
            ));
        }
        let table_name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        pos += name_len;

        // Segments: V6 fixed = 56 bytes/segment, V7 fixed = 64
        // bytes/segment (the extra 8 are `visible_at_lsn`). Both end
        // with a variable-length file_path (u32 len + utf-8 bytes).
        let seg_fixed_len = if is_v7 { 64 } else { 56 };
        let seg_count = read_u32(data, &mut pos)? as usize;
        let mut segments = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            if pos + seg_fixed_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "manifest truncated at segment",
                ));
            }
            let segment_id = read_u64(data, &mut pos)?;
            let row_count = read_u64(data, &mut pos)? as usize;
            let min_row_id = read_i64(data, &mut pos)?;
            let max_row_id = read_i64(data, &mut pos)?;
            let creation_lsn = read_u64(data, &mut pos)?;
            let seal_seq = read_u64(data, &mut pos)?;
            let schema_version = read_u64(data, &mut pos)?;
            // V7 carries visible_at_lsn explicitly. V6 doesn't, so
            // synthesize from creation_lsn — V6 segments don't have
            // a precise SWMR-visible LSN, but the writer's checkpoint
            // ordering guarantees that any reader who could see
            // creation_lsn (which is bounded by checkpoint_lsn) could
            // also have seen this segment.
            let visible_at_lsn = if is_v7 {
                read_u64(data, &mut pos)?
            } else {
                creation_lsn
            };

            let path_len = read_u32(data, &mut pos)? as usize;
            if pos + path_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "manifest truncated at path",
                ));
            }
            let path_str = std::str::from_utf8(&data[pos..pos + path_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            pos += path_len;

            segments.push(SegmentMeta {
                segment_id,
                file_path: PathBuf::from(path_str),
                row_count,
                min_row_id,
                max_row_id,
                creation_lsn,
                seal_seq,
                schema_version,
                visible_at_lsn,
            });
        }

        // Last 4 bytes are CRC, so stop before them.
        let data_end = data.len() - 4;

        // Tombstones: V7 = (row_id, commit_seq, visible_at_lsn)
        // 24-byte triples; V6 = (row_id, commit_seq) 16-byte pairs
        // with synthesized visible_at_lsn=0 ("always visible" — V6
        // pre-dates capped read-only attaches).
        let mut tombstones: Vec<(i64, u64, u64)> = Vec::new();
        if pos + 8 <= data_end {
            let tombstone_count = read_u64(data, &mut pos)? as usize;
            let entry_size = if tombstone_has_visible_lsn { 24 } else { 16 };
            tombstones.reserve(tombstone_count);
            for _ in 0..tombstone_count {
                if pos + entry_size > data_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest truncated at tombstone entry",
                    ));
                }
                let row_id = read_i64(data, &mut pos)?;
                let commit_seq = read_u64(data, &mut pos)?;
                let visible_at_lsn = if tombstone_has_visible_lsn {
                    read_u64(data, &mut pos)?
                } else {
                    0
                };
                tombstones.push((row_id, commit_seq, visible_at_lsn));
            }
        }

        // Column renames: (old_name, new_name) pairs
        let mut column_renames = Vec::new();
        if pos + 4 <= data_end {
            let rename_count = read_u32(data, &mut pos)? as usize;
            column_renames.reserve(rename_count);
            for _ in 0..rename_count {
                if pos + 2 > data_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest truncated at column rename entry",
                    ));
                }
                let old_len = u16::from_le_bytes(data[pos..pos + 2].try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "truncated rename old_name len")
                })?) as usize;
                pos += 2;
                if pos + old_len > data_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest truncated at column rename old_name",
                    ));
                }
                let old_name = std::str::from_utf8(&data[pos..pos + old_len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                pos += old_len;
                if pos + 2 > data_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest truncated at column rename new_name len",
                    ));
                }
                let new_len = u16::from_le_bytes(data[pos..pos + 2].try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "truncated rename new_name len")
                })?) as usize;
                pos += 2;
                if pos + new_len > data_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest truncated at column rename new_name",
                    ));
                }
                let new_name = std::str::from_utf8(&data[pos..pos + new_len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                pos += new_len;
                column_renames.push((SmartString::from(old_name), SmartString::from(new_name)));
            }
        }

        // Dropped columns: (name, schema_version) pairs
        let mut dropped_columns = Vec::new();
        if pos + 4 <= data_end {
            let count = read_u32(data, &mut pos)? as usize;
            for _ in 0..count {
                if pos + 2 > data_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest truncated at dropped column entry",
                    ));
                }
                let nlen = u16::from_le_bytes(data[pos..pos + 2].try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "truncated dropped col name")
                })?) as usize;
                pos += 2;
                if pos + nlen > data_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest truncated at dropped column name",
                    ));
                }
                let name = std::str::from_utf8(&data[pos..pos + nlen])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                pos += nlen;
                let drop_version = read_u64(data, &mut pos)?;
                dropped_columns.push((SmartString::from(name), drop_version));
            }
        }

        Ok(Self {
            table_name: SmartString::from(table_name),
            segments,
            next_segment_id,
            checkpoint_lsn,
            tombstones,
            column_renames,
            dropped_columns,
            schema_version: table_schema_version,
        })
    }

    /// Write manifest to disk atomically.
    pub fn write_to_disk(&self, path: &Path) -> Result<()> {
        let data = self.serialize().map_err(|e| {
            crate::core::Error::internal(format!("failed to serialize manifest: {}", e))
        })?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                crate::core::Error::internal(format!("failed to create manifest dir: {}", e))
            })?;
        }

        // Write to tmp file and fsync BEFORE rename for crash safety.
        let tmp_path = path.with_extension("manifest.tmp");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp_path).map_err(|e| {
                crate::core::Error::internal(format!("failed to create manifest tmp file: {}", e))
            })?;
            f.write_all(&data).map_err(|e| {
                crate::core::Error::internal(format!("failed to write manifest: {}", e))
            })?;
            f.sync_all().map_err(|e| {
                crate::core::Error::internal(format!("failed to fsync manifest: {}", e))
            })?;
        }
        std::fs::rename(&tmp_path, path).map_err(|e| {
            crate::core::Error::internal(format!("failed to rename manifest: {}", e))
        })?;
        // Fsync parent directory to ensure the rename is durable.
        // Windows does not support opening directories for fsync;
        // NTFS metadata is flushed with the file's sync_all().
        #[cfg(not(windows))]
        if let Some(parent) = path.parent() {
            let d = std::fs::File::open(parent).map_err(|e| {
                crate::core::Error::internal(format!("failed to open dir for fsync: {}", e))
            })?;
            d.sync_all().map_err(|e| {
                crate::core::Error::internal(format!("failed to fsync manifest dir: {}", e))
            })?;
        }
        Ok(())
    }

    /// Read manifest from disk.
    pub fn read_from_disk(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| crate::core::Error::internal(format!("failed to read manifest: {}", e)))?;
        Self::deserialize(&data).map_err(|e| {
            crate::core::Error::internal(format!("failed to deserialize manifest: {}", e))
        })
    }
}

/// Recompute the visibility bitmaps for all segments in `segments`.
///
/// `seg_order` lists segment IDs in ascending (oldest-first) order.
/// Segments are processed newest-first: the first time a row_id is seen it is
/// marked visible; subsequent occurrences (in older volumes) are masked out.
/// When there is at most one segment every row is authoritative, so all
/// `visible` fields are set to `None` (fast path for the common case).
fn compute_visibility_bitmaps(
    seg_order: &[u64],
    segments: &mut rustc_hash::FxHashMap<u64, ColdSegment>,
    reusable_seen: &mut rustc_hash::FxHashSet<i64>,
) {
    if segments.len() <= 1 {
        for cs in segments.values_mut() {
            cs.visible = None;
        }
        return;
    }
    // Reuse the caller's seen set — clear and resize, but keep the allocation.
    reusable_seen.clear();
    let total: usize = segments.values().map(|cs| cs.volume.meta.row_count).sum();
    if reusable_seen.capacity() < total {
        reusable_seen.reserve(total * 8 / 7 + 16 - reusable_seen.capacity());
    }
    // Process newest-first (seg_order is ascending, so iterate reversed)
    for &seg_id in seg_order.iter().rev() {
        if let Some(cs) = segments.get_mut(&seg_id) {
            let rc = cs.volume.meta.row_count;
            if rc == 0 {
                cs.visible = None;
                continue;
            }
            let num_words = rc.div_ceil(64);
            let mut bits = vec![!0u64; num_words];
            // Clear trailing bits beyond row_count
            let trailing = rc % 64;
            if trailing != 0 {
                bits[num_words - 1] &= (1u64 << trailing) - 1;
            }
            let mut has_overlap = false;
            for i in 0..rc {
                if !reusable_seen.insert(cs.volume.meta.row_ids[i]) {
                    bits[i >> 6] &= !(1u64 << (i & 63));
                    has_overlap = true;
                }
            }
            // No overlap with newer volumes: None = all visible (zero memory, no per-row check)
            cs.visible = if has_overlap {
                Some(Arc::new(bits))
            } else {
                None
            };
        }
    }
    // Shrink if capacity far exceeds what was needed. After compaction
    // merges volumes, the total row count drops but the set stays at its
    // high-water mark. Replace with a right-sized set to free the excess.
    if reusable_seen.capacity() > total * 2 + 1024 {
        *reusable_seen = rustc_hash::FxHashSet::with_capacity_and_hasher(total, Default::default());
    } else {
        reusable_seen.clear();
    }
}

/// Per-table segment manager.
///
/// Owns the manifest, loaded segments, and tombstone set for one table.
/// Tombstones track cold row_ids that have been deleted or superseded
/// by hot buffer versions. They are persisted in the manifest and used
/// during scans to skip stale cold rows.
///
/// Thread safety: the manager uses interior mutability via RwLock for
/// concurrent read access (queries) and exclusive write access (seal, compaction).
pub struct SegmentManager {
    /// Table name.
    table_name: SmartString,
    /// The manifest (source of truth for segment state).
    manifest: RwLock<TableManifest>,
    /// Loaded segments with pre-computed column mappings, keyed by segment_id.
    /// CoW via Arc: readers clone the Arc (O(1) atomic increment, ~5ns),
    /// writers clone the inner map, modify, and swap the Arc.
    /// The ColumnMapping is computed once at registration and recomputed on ALTER TABLE.
    /// This eliminates per-scan compute_column_mapping overhead and lock contention.
    segments: RwLock<Arc<FxHashMap<u64, ColdSegment>>>,
    /// Base directory for volume files (None for memory-only databases).
    volume_dir: Option<PathBuf>,
    /// Fast atomic flag: true if any segments are loaded.
    has_segments_flag: std::sync::atomic::AtomicBool,
    /// Current eviction epoch. Updated by evict_idle_volumes.
    pub current_eviction_epoch: std::sync::atomic::AtomicU64,
    /// True when any segment in the map is cold (metadata only, needs reload).
    has_cold: std::sync::atomic::AtomicBool,
    /// Serializes reload attempts. Concurrent callers block on this mutex
    /// instead of spinning, preventing CPU waste during disk I/O.
    reloading: parking_lot::Mutex<()>,
    /// Committed tombstone map: cold row_id → commit_seq (when the tombstone was created).
    /// Built from manifest tombstones on startup, updated at commit time.
    /// Wrapped in Arc for cheap O(1) reads — most callers only need to check
    /// membership, not mutate. Writers swap the Arc on mutation.
    /// The commit_seq enables snapshot isolation: a snapshot at begin_seq=N
    /// only sees tombstones with commit_seq <= N.
    tombstones: RwLock<Arc<FxHashMap<i64, u64>>>,
    /// Per-transaction pending tombstones: txn_id → list of cold row_ids to tombstone.
    /// Applied to the shared tombstone set on commit, discarded on rollback.
    /// This lives on the SegmentManager (not SegmentedTable) because the commit
    /// path in engine.rs creates fresh MVCCTable instances that don't have
    /// access to SegmentedTable state.
    pending_txn_tombstones: RwLock<FxHashMap<i64, FxHashSet<i64>>>,
    // Unique constraint checks use per-volume hash indices (on FrozenVolume).
    // No global cache needed. Each volume builds its index lazily on first
    // unique check and never invalidates (volumes are immutable).
    // Zone maps + bloom filters prune volumes before hash lookup.
    /// Cached deduplicated row count. Invalidated (set to u64::MAX) on
    /// segment or tombstone changes. Recomputed lazily on next read.
    cached_deduped_count: std::sync::atomic::AtomicU64,
    /// Per-table fence that serializes seal with cold-check + hot insert.
    /// INSERTs take a shared guard while checking cold constraints and
    /// publishing into hot; seal takes the exclusive guard while moving rows.
    seal_fence: RwLock<()>,
    /// Reusable scratch set for compute_visibility_bitmaps. Kept alive across
    /// calls to avoid re-allocating 200 MB on every seal/compact. Protected by
    /// Mutex since visibility computation is always single-threaded (under
    /// segments write lock).
    visibility_seen: parking_lot::Mutex<rustc_hash::FxHashSet<i64>>,
    /// Monotonic counter incremented on every register_segment. Used at
    /// commit time to detect whether a seal happened since statement time.
    /// If unchanged, the commit-time cold recheck is skipped (fast path).
    seal_generation: std::sync::atomic::AtomicU64,
    /// Per-txn seal generation at INSERT time. Small map — only active
    /// transactions with pending inserts on this table.
    txn_seal_gens: parking_lot::Mutex<rustc_hash::FxHashMap<i64, u64>>,
    /// Number of rows currently being sealed (exist in both hot and cold).
    /// Set to N before register_segment, cleared after remove_sealed_rows.
    /// Subtracted from row_count() to prevent double-counting during the seal window.
    seal_overlap_count: std::sync::atomic::AtomicUsize,
}

impl SegmentManager {
    /// Create a new segment manager for a table.
    pub fn new(table_name: &str, volume_dir: Option<PathBuf>) -> Self {
        Self {
            table_name: SmartString::from(table_name),
            manifest: RwLock::new(TableManifest::new(table_name)),
            segments: RwLock::new(Arc::new(FxHashMap::default())),
            volume_dir,
            has_segments_flag: std::sync::atomic::AtomicBool::new(false),
            has_cold: std::sync::atomic::AtomicBool::new(false),
            current_eviction_epoch: std::sync::atomic::AtomicU64::new(0),
            reloading: parking_lot::Mutex::new(()),
            tombstones: RwLock::new(Arc::new(FxHashMap::default())),
            pending_txn_tombstones: RwLock::new(FxHashMap::default()),
            cached_deduped_count: std::sync::atomic::AtomicU64::new(u64::MAX),
            seal_fence: RwLock::new(()),
            visibility_seen: parking_lot::Mutex::new(rustc_hash::FxHashSet::default()),
            seal_generation: std::sync::atomic::AtomicU64::new(0),
            txn_seal_gens: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            seal_overlap_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create from an existing manifest loaded from disk.
    pub fn from_manifest(manifest: TableManifest, volume_dir: Option<PathBuf>) -> Self {
        let table_name = manifest.table_name.clone();
        let tombstone_map: FxHashMap<i64, u64> = manifest
            .tombstones
            .iter()
            .map(|&(rid, seq, _vis)| (rid, seq))
            .collect();
        Self {
            table_name,
            manifest: RwLock::new(manifest),
            segments: RwLock::new(Arc::new(FxHashMap::default())),
            volume_dir,
            has_segments_flag: std::sync::atomic::AtomicBool::new(false),
            has_cold: std::sync::atomic::AtomicBool::new(false),
            current_eviction_epoch: std::sync::atomic::AtomicU64::new(0),
            reloading: parking_lot::Mutex::new(()),
            tombstones: RwLock::new(Arc::new(tombstone_map)),
            pending_txn_tombstones: RwLock::new(FxHashMap::default()),
            cached_deduped_count: std::sync::atomic::AtomicU64::new(u64::MAX),
            seal_fence: RwLock::new(()),
            visibility_seen: parking_lot::Mutex::new(rustc_hash::FxHashSet::default()),
            seal_generation: std::sync::atomic::AtomicU64::new(0),
            txn_seal_gens: parking_lot::Mutex::new(rustc_hash::FxHashMap::default()),
            seal_overlap_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get the table name.
    pub fn table_name(&self) -> &str {
        &self.table_name
    }

    /// Ensure all volumes have column data before column access.
    /// Reloads cold segments (in map with metadata only, columns missing).
    fn ensure_columns(&self) {
        if !self.has_cold.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let _guard = self.reloading.lock();
        if !self.has_cold.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let cold_ids: Vec<u64> = {
            let segs = self.segments.read();
            segs.iter()
                .filter(|(_, cs)| cs.volume.is_cold())
                .map(|(&id, _)| id)
                .collect()
        };
        if !cold_ids.is_empty() {
            self.reload_cold_volumes(cold_ids);
        }
    }

    /// Get segments in order, metadata only (no cold volume reload).
    /// Use when callers only need vol.meta (stats, zone maps, row_ids).
    /// Does NOT mark volumes as accessed — metadata reads should not
    /// prevent eviction of column data.
    pub fn get_segments_ordered_meta(&self) -> Vec<Arc<FrozenVolume>> {
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        seg_ids
            .iter()
            .filter_map(|id| segments.get(id).map(|cs| Arc::clone(&cs.volume)))
            .collect()
    }

    /// Get volumes in newest-first order (by segment_id, descending).
    /// Used for building per-volume skip sets in SegmentedTable.scan().
    /// Segments are always appended in ascending order, so reverse gives
    /// newest-first in O(n) instead of O(n log n) sort.
    pub fn get_volumes_newest_first(&self) -> Arc<Vec<(u64, ColdSegment)>> {
        self.ensure_columns();
        let mut result = self.build_volumes_newest_first();
        // Race check: eviction may have created cold volumes between
        // ensure_columns (fast-path has_cold=false) and segments.read().
        // Retry once — eviction runs once per ~30s checkpoint cycle.
        if result.iter().any(|(_, cs)| cs.volume.is_cold()) {
            self.ensure_columns();
            result = self.build_volumes_newest_first();
            // If still cold after retry (persistent reload failure),
            // filter them out to prevent column-access panics.
            let cold: Vec<(u64, usize)> = result
                .iter()
                .filter(|(_, cs)| cs.volume.is_cold())
                .map(|(id, cs)| (*id, cs.volume.meta.row_count))
                .collect();
            if !cold.is_empty() {
                for &(seg_id, rows) in &cold {
                    eprintln!(
                        "Warning: table {} seg={}: cold volume excluded ({} rows, reload failed)",
                        self.table_name, seg_id, rows
                    );
                }
                result.retain(|(_, cs)| !cs.volume.is_cold());
            }
        }
        // Mark all volumes accessed. Direct-iteration callers (aggregate
        // pushdown, DML) access column data without zone-map pruning, so
        // they need protection from eviction. Scanner paths that prune
        // first use get_volumes_newest_first_lazy() instead.
        for (_, cs) in &result {
            cs.volume.mark_accessed();
        }
        result.reverse();
        Arc::new(result)
    }

    /// Build the raw newest-first volume list from manifest + segments.
    fn build_volumes_newest_first(&self) -> Vec<(u64, ColdSegment)> {
        let (seg_ids, segs) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let segs = Arc::clone(&*self.segments.read());
            (seg_ids, segs)
        };
        seg_ids
            .iter()
            .filter_map(|&id| segs.get(&id).map(|cs| (id, cs.clone())))
            .collect()
    }

    /// Same as get_volumes_newest_first but without ensure_columns.
    /// Used by scanner paths that prune by zone maps/bloom filters before
    /// accessing column data. Cold volumes are loaded on demand via
    /// ensure_volume after pruning, avoiding full cold-set reload.
    /// Does NOT mark volumes — only volumes that survive pruning get marked
    /// by the scanner constructor or explicit per-volume mark_accessed.
    pub fn get_volumes_newest_first_lazy(&self) -> Arc<Vec<(u64, ColdSegment)>> {
        let (seg_ids, segs) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let segs = Arc::clone(&*self.segments.read());
            (seg_ids, segs)
        };
        let mut result: Vec<(u64, ColdSegment)> = seg_ids
            .iter()
            .filter_map(|&id| segs.get(&id).map(|cs| (id, cs.clone())))
            .collect();
        result.reverse();
        Arc::new(result)
    }

    /// Atomic snapshot of newest-first volumes AND the tombstone map
    /// that applies to them. Use this in any scan path that needs both
    /// pieces of cold state — calling `get_volumes_newest_first()` and
    /// `tombstone_set_arc()` separately would race against a
    /// concurrent `reload_from_disk` and yield a mixed snapshot
    /// (e.g. old volumes with new tombstones), silently resurrecting
    /// stale rows or hiding live ones.
    ///
    /// Atomicity is provided by holding `manifest.read()` across the
    /// segments + tombstones reads. `reload_from_disk` acquires
    /// `manifest.write()` FIRST in its three-lock publish, so as long
    /// as our `manifest.read()` is held, none of the three Arcs can
    /// be swapped underneath us.
    pub fn volumes_and_tombstones_newest_first(&self) -> (SharedSegmentList, TombstoneMap) {
        self.ensure_columns();
        let mut result = self.build_volumes_and_tombstones();
        // Race check: same as `get_volumes_newest_first`. Eviction may
        // have created cold volumes between `ensure_columns` (fast-path
        // `has_cold=false`) and our snapshot. Retry once.
        if result.0.iter().any(|(_, cs)| cs.volume.is_cold()) {
            self.ensure_columns();
            result = self.build_volumes_and_tombstones();
            let cold: Vec<(u64, usize)> = result
                .0
                .iter()
                .filter(|(_, cs)| cs.volume.is_cold())
                .map(|(id, cs)| (*id, cs.volume.meta.row_count))
                .collect();
            if !cold.is_empty() {
                for &(seg_id, rows) in &cold {
                    eprintln!(
                        "Warning: table {} seg={}: cold volume excluded ({} rows, reload failed)",
                        self.table_name, seg_id, rows
                    );
                }
                let (mut vols, ts) = result;
                vols.retain(|(_, cs)| !cs.volume.is_cold());
                result = (vols, ts);
            }
        }
        let (mut volumes, ts) = result;
        for (_, cs) in &volumes {
            cs.volume.mark_accessed();
        }
        volumes.reverse();
        (Arc::new(volumes), ts)
    }

    /// Internal helper for `volumes_and_tombstones_newest_first`. Returns
    /// the raw (oldest-first) volume list and the tombstone Arc captured
    /// under a single `manifest.read()`.
    fn build_volumes_and_tombstones(&self) -> (SegmentList, TombstoneMap) {
        let manifest = self.manifest.read();
        let segs = Arc::clone(&*self.segments.read());
        let ts = Arc::clone(&*self.tombstones.read());
        let result: Vec<(u64, ColdSegment)> = manifest
            .segments
            .iter()
            .filter_map(|m| segs.get(&m.segment_id).map(|cs| (m.segment_id, cs.clone())))
            .collect();
        (result, ts)
    }

    /// Atomic lazy variant of `volumes_and_tombstones_newest_first`. Same
    /// race-free guarantee, but skips `ensure_columns` (scanner paths
    /// load surviving volumes on demand via `ensure_volume` after
    /// zone-map/bloom pruning).
    pub fn volumes_and_tombstones_newest_first_lazy(&self) -> (SharedSegmentList, TombstoneMap) {
        let (mut volumes, ts) = self.build_volumes_and_tombstones();
        volumes.reverse();
        (Arc::new(volumes), ts)
    }

    /// Check if there are any segments. O(1) atomic read, no lock.
    pub fn has_segments(&self) -> bool {
        self.has_segments_flag
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Capture an atomic snapshot of segment state for batch constraint checking.
    /// Acquires manifest + segments + tombstones locks once. All per-row checks
    /// within the batch reuse this snapshot with zero lock overhead.
    pub fn cold_snapshot(&self) -> ColdSnapshot {
        // No ensure_columns — zone-map/bloom pruning uses metadata (available
        // on cold volumes). Only volumes that pass pruning get loaded on demand
        // via ensure_volume in check_value_exists_impl/find_row_id_by_values_impl.
        let manifest = self.manifest.read();
        let seg_ids: smallvec::SmallVec<[u64; 4]> = manifest
            .segments
            .iter()
            .rev()
            .map(|m| m.segment_id)
            .collect();
        let segs = Arc::clone(&*self.segments.read());
        let ts = Arc::clone(&*self.tombstones.read());
        drop(manifest);
        ColdSnapshot { seg_ids, segs, ts }
    }

    /// Check if a value exists using a pre-captured snapshot (no lock acquisition).
    pub fn check_value_exists_with_snapshot(
        &self,
        snapshot: &ColdSnapshot,
        col_idx: usize,
        value: &crate::core::Value,
    ) -> Option<i64> {
        self.check_value_exists_impl(
            &snapshot.seg_ids,
            &snapshot.segs,
            &snapshot.ts,
            col_idx,
            value,
        )
    }

    /// Check if a value exists in cold segments (acquires locks per call).
    /// For batch operations, prefer cold_snapshot() + check_value_exists_with_snapshot().
    pub fn check_value_exists_in_segments(
        &self,
        col_idx: usize,
        value: &crate::core::Value,
    ) -> Option<i64> {
        let snapshot = self.cold_snapshot();
        self.check_value_exists_impl(
            &snapshot.seg_ids,
            &snapshot.segs,
            &snapshot.ts,
            col_idx,
            value,
        )
    }

    /// Resolve the current value at a logical column for a given row_id.
    /// Iterates newest-first with per-volume physical mapping. Used by
    /// overlap verification to check the authoritative version after
    /// UPDATE changes a PK value + seal (schema-evolution safe).
    fn get_authoritative_value(&self, row_id: i64, col_idx: usize) -> Option<crate::core::Value> {
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| m.segment_id)
                .collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        for seg_id in &seg_ids {
            if let Some(cold) = segments.get(seg_id) {
                if let Ok(idx) = cold.volume.meta.row_ids.binary_search(&row_id) {
                    let pi = if cold.mapping.is_identity {
                        col_idx
                    } else if col_idx < cold.mapping.sources.len() {
                        match &cold.mapping.sources[col_idx] {
                            super::writer::ColSource::Volume(vi) => *vi,
                            super::writer::ColSource::Default(val) => return Some(val.clone()),
                        }
                    } else {
                        return None;
                    };
                    if cold.volume.is_cold() {
                        if let Some(vol) = self.ensure_volume(*seg_id) {
                            return Some(vol.columns[pi].get_value(idx));
                        }
                        return None;
                    }
                    cold.volume.mark_accessed();
                    return Some(cold.volume.columns[pi].get_value(idx));
                }
            }
        }
        None
    }

    fn check_value_exists_impl(
        &self,
        seg_ids: &[u64],
        segs: &FxHashMap<u64, ColdSegment>,
        ts: &FxHashMap<i64, u64>,
        col_idx: usize,
        value: &crate::core::Value,
    ) -> Option<i64> {
        let mut seen = FxHashSet::default();

        for &seg_id in seg_ids {
            let Some(cold) = segs.get(&seg_id) else {
                continue;
            };
            let vol = &cold.volume;
            // Resolve logical col_idx to physical index via per-volume mapping.
            // After DROP COLUMN, older volumes may store the column at a
            // different position than the current schema ordinal.
            let pi = if cold.mapping.is_identity {
                if col_idx >= vol.columns.len() {
                    continue;
                }
                col_idx
            } else if col_idx < cold.mapping.sources.len() {
                match &cold.mapping.sources[col_idx] {
                    super::writer::ColSource::Volume(vi) => *vi,
                    super::writer::ColSource::Default(_) => continue,
                }
            } else {
                continue;
            };
            if pi >= vol.meta.zone_maps.len() || !vol.meta.zone_maps[pi].may_contain_eq(value) {
                continue;
            }
            // Zone map passed — need column data. Load cold volumes on demand.
            let loaded: Arc<FrozenVolume>;
            let vol = if vol.is_cold() {
                loaded = match self.ensure_volume(seg_id) {
                    Some(v) => v,
                    None => continue,
                };
                &*loaded
            } else {
                vol.mark_accessed();
                vol
            };
            let target = match value {
                crate::core::Value::Integer(int_val) => Some(*int_val),
                crate::core::Value::Timestamp(ts_val) => Some(
                    ts_val
                        .timestamp_nanos_opt()
                        .unwrap_or(ts_val.timestamp() * 1_000_000_000),
                ),
                _ => None,
            };
            if let Some(target) = target {
                if vol.is_sorted(pi) {
                    let start = vol.columns[pi].binary_search_ge(target);
                    let mut i = start;
                    while i < vol.meta.row_count && vol.columns[pi].get_i64(i) == target {
                        let rid = vol.meta.row_ids[i];
                        if seen.insert(rid) && !ts.contains_key(&rid) {
                            if seg_ids.len() > 1 {
                                if let Some(current_val) =
                                    self.get_authoritative_value(rid, col_idx)
                                {
                                    if &current_val != value {
                                        i += 1;
                                        continue;
                                    }
                                }
                            }
                            return Some(rid);
                        }
                        i += 1;
                    }
                } else {
                    for i in 0..vol.meta.row_count {
                        let rid = vol.meta.row_ids[i];
                        if !seen.insert(rid) {
                            continue;
                        }
                        if !vol.columns[pi].is_null(i)
                            && vol.columns[pi].get_i64(i) == target
                            && !ts.contains_key(&rid)
                        {
                            if seg_ids.len() > 1 {
                                if let Some(current_val) =
                                    self.get_authoritative_value(rid, col_idx)
                                {
                                    if &current_val != value {
                                        continue;
                                    }
                                }
                            }
                            return Some(rid);
                        }
                    }
                }
            }
        }
        None
    }

    /// Find a visible cold row ID matching the given column values.
    /// Uses a three-tier pruning strategy per volume (newest first):
    ///   1. Zone map: skip if value outside [min, max] for any column
    ///   2. Bloom filter: skip if any column says "definitely not"
    ///   3. Per-volume hash index: O(1) lookup (lazily built, never invalidated)
    ///
    /// No global cache. Each volume's hash index is built once on first use
    /// and lives on the immutable FrozenVolume. Zero invalidation cost.
    /// Find row by values using a pre-captured snapshot (no lock acquisition).
    pub fn find_row_id_by_values_with_snapshot(
        &self,
        snapshot: &ColdSnapshot,
        col_indices: &[usize],
        values: &[&Value],
        column_defaults: &[Value],
    ) -> Option<i64> {
        self.find_row_id_by_values_impl(
            &snapshot.seg_ids,
            &snapshot.segs,
            &snapshot.ts,
            col_indices,
            values,
            column_defaults,
        )
    }

    pub fn find_row_id_by_values(
        &self,
        col_indices: &[usize],
        values: &[&Value],
        column_defaults: &[Value],
    ) -> Option<i64> {
        let snapshot = self.cold_snapshot();
        self.find_row_id_by_values_impl(
            &snapshot.seg_ids,
            &snapshot.segs,
            &snapshot.ts,
            col_indices,
            values,
            column_defaults,
        )
    }

    fn find_row_id_by_values_impl(
        &self,
        seg_ids: &[u64],
        segs: &FxHashMap<u64, ColdSegment>,
        ts: &FxHashMap<i64, u64>,
        col_indices: &[usize],
        values: &[&Value],
        column_defaults: &[Value],
    ) -> Option<i64> {
        if col_indices.is_empty() || col_indices.len() != values.len() {
            return None;
        }

        if seg_ids.is_empty() {
            return None;
        }

        let bloom_hashes: smallvec::SmallVec<[u64; 4]> = values
            .iter()
            .map(|v| super::column::ColumnBloomFilter::hash_value_static(v))
            .collect();

        // Track seen row_ids for newest-first dedup across overlapping volumes.
        let mut seen = FxHashSet::default();

        for &seg_id in seg_ids {
            let Some(cold) = segs.get(&seg_id) else {
                continue;
            };
            let vol = &cold.volume;
            // Derive remap from ColdSegment.mapping (already cached per volume).
            // Maps schema column indices to volume column indices.
            // Missing columns (ColSource::Default) are usize::MAX.
            let mut vol_col_indices: smallvec::SmallVec<[usize; 4]> =
                smallvec::SmallVec::with_capacity(col_indices.len());
            let mut has_missing = false;
            let mut skip_vol = false;
            for (i, &ci) in col_indices.iter().enumerate() {
                if ci < cold.mapping.sources.len() {
                    match &cold.mapping.sources[ci] {
                        super::writer::ColSource::Volume(vi) => {
                            vol_col_indices.push(*vi);
                        }
                        super::writer::ColSource::Default(_) => {
                            has_missing = true;
                            // Column missing from volume. If searched value
                            // doesn't match default, no row can match.
                            if *values[i] != column_defaults[i] {
                                skip_vol = true;
                                break;
                            }
                            vol_col_indices.push(usize::MAX);
                        }
                    }
                } else {
                    // Schema column index out of range for this mapping
                    has_missing = true;
                    if *values[i] != column_defaults[i] {
                        skip_vol = true;
                        break;
                    }
                    vol_col_indices.push(usize::MAX);
                }
            }
            if skip_vol {
                continue;
            }

            // Tier 1: Zone map pruning
            let mut zone_skip = false;
            for (i, &vi) in vol_col_indices.iter().enumerate() {
                if vi < vol.meta.zone_maps.len()
                    && !vol.meta.zone_maps[vi].may_contain_eq(values[i])
                {
                    zone_skip = true;
                    break;
                }
            }
            if zone_skip {
                continue;
            }

            // Tier 2: Bloom filter pruning
            let mut bloom_skip = false;
            for (i, &vi) in vol_col_indices.iter().enumerate() {
                if vi < vol.meta.bloom_filters.len()
                    && !vol.meta.bloom_filters[vi].might_contain_hash(bloom_hashes[i])
                {
                    bloom_skip = true;
                    break;
                }
            }
            if bloom_skip {
                continue;
            }

            // Zone map + bloom passed — need column data. Load cold on demand.
            let loaded: Arc<FrozenVolume>;
            let vol = if vol.is_cold() {
                loaded = match self.ensure_volume(seg_id) {
                    Some(v) => v,
                    None => continue,
                };
                &*loaded
            } else {
                vol.mark_accessed();
                vol
            };
            // Tier 3: Per-volume hash index
            let mut vol_result: Option<i64> = None;
            if !has_missing {
                // Common path: no schema evolution, pass values directly (zero alloc)
                vol.unique_lookup_all(&vol_col_indices, values, |row_idx| {
                    let rid = vol.meta.row_ids[row_idx as usize];
                    if ts.contains_key(&rid) {
                        false
                    } else if seen.insert(rid) {
                        vol_result = Some(rid);
                        true
                    } else {
                        false
                    }
                });
            } else {
                // Schema-evolved volume: some columns missing (default matches).
                // Check only the columns that exist in the volume.
                let present_cols: Vec<(usize, usize)> = vol_col_indices
                    .iter()
                    .enumerate()
                    .filter(|(_, &vi)| vi != usize::MAX)
                    .map(|(i, &vi)| (i, vi))
                    .collect();
                for i in 0..vol.meta.row_count {
                    let rid = vol.meta.row_ids[i];
                    if ts.contains_key(&rid) || !seen.insert(rid) {
                        continue;
                    }
                    let matches = present_cols.iter().all(|&(val_idx, ci)| {
                        let v = vol.columns[ci].get_value(i);
                        !v.is_null() && v == *values[val_idx]
                    });
                    if matches {
                        vol_result = Some(rid);
                        break;
                    }
                }
            }
            if let Some(rid) = vol_result {
                // Verify this is the authoritative version. After UPDATE old→new
                // + seal, overlapping volumes can have the same row_id with
                // different values. The older volume's stale value is not
                // tombstoned (tombstone cleared when row_id appeared in the newer
                // volume). get_cold_row returns the newest version (newest-first).
                // Use column_defaults for columns missing from schema-evolved volumes.
                if seg_ids.len() > 1 {
                    let still_matches = col_indices.iter().enumerate().all(|(i, &ci)| {
                        if let Some(v) = self.get_authoritative_value(rid, ci) {
                            !v.is_null() && v == *values[i]
                        } else {
                            column_defaults[i] == *values[i]
                        }
                    });
                    if !still_matches {
                        continue; // stale value in older volume, skip
                    }
                }
                return Some(rid);
            }
        }
        None
    }

    /// Get the number of segments.
    pub fn segment_count(&self) -> usize {
        self.manifest.read().segments.len()
    }

    /// Get the number of committed tombstones.
    pub fn tombstone_count(&self) -> usize {
        self.tombstones.read().len()
    }

    /// Get the maximum row_count across all segments.
    pub fn max_segment_row_count(&self) -> usize {
        let manifest = self.manifest.read();
        manifest
            .segments
            .iter()
            .map(|s| s.row_count)
            .max()
            .unwrap_or(0)
    }

    /// Per-volume statistics for PRAGMA VOLUME_STATS.
    /// Returns (segment_id, tier, row_count, memory_bytes, idle_cycles) for each volume.
    pub fn volume_stats(&self) -> Vec<(u64, &'static str, usize, usize, u64)> {
        let current_epoch = self
            .current_eviction_epoch
            .load(std::sync::atomic::Ordering::Relaxed);
        let manifest = self.manifest.read();
        let segments = self.segments.read();
        let mut stats = Vec::with_capacity(manifest.segments.len());
        for meta in &manifest.segments {
            let seg_id = meta.segment_id;
            if let Some(cs) = segments.get(&seg_id) {
                let vol = &cs.volume;
                let tier = if vol.columns.is_eager() {
                    "hot"
                } else if vol.columns.has_compressed_store() {
                    "warm"
                } else {
                    "cold"
                };
                let row_count = vol.meta.row_ids.len();
                let memory_bytes = vol.memory_size();
                let last_epoch = vol
                    .last_access_epoch
                    .load(std::sync::atomic::Ordering::Relaxed);
                let idle_cycles = if last_epoch == u64::MAX || current_epoch == 0 {
                    0
                } else {
                    current_epoch.saturating_sub(last_epoch)
                };
                stats.push((seg_id, tier, row_count, memory_bytes, idle_cycles));
            }
        }
        stats
    }

    /// Evict idle volumes to save memory. Three-tier transitions:
    ///
    /// - Hot → Warm (drop decompressed columns, keep compressed blocks in RAM)
    /// - Warm → Cold (drop compressed blocks, remove from map, track for disk reload)
    ///
    /// Volumes must be idle for MIN_IDLE_CYCLES before each transition.
    /// Metadata is shared via Arc, zero allocation for hot→warm and warm→cold.
    pub fn evict_idle_volumes(&self, current_epoch: u64) {
        const MIN_IDLE_CYCLES: u64 = 3;

        // Publish current epoch so scanners can stamp volumes correctly.
        self.current_eviction_epoch
            .store(current_epoch, std::sync::atomic::Ordering::Relaxed);

        // Identify targets under read lock. Reset accessed volumes' epochs
        // so their idle counter starts from this cycle.
        let mut has_targets = false;
        let targets: Vec<(u64, bool, bool)> = {
            let segs = self.segments.read();
            segs.iter()
                .filter_map(|(&seg_id, cs)| {
                    let vol_epoch = cs
                        .volume
                        .last_access_epoch
                        .load(std::sync::atomic::Ordering::Relaxed);
                    if vol_epoch == u64::MAX {
                        cs.volume
                            .last_access_epoch
                            .store(current_epoch, std::sync::atomic::Ordering::Relaxed);
                        return None;
                    }
                    let delta = current_epoch.saturating_sub(vol_epoch);
                    if delta < MIN_IDLE_CYCLES {
                        return None;
                    }
                    let is_hot = cs.volume.columns.is_eager();
                    let is_warm = cs.volume.is_warm();
                    if is_hot || is_warm {
                        has_targets = true;
                        Some((seg_id, is_hot, is_warm))
                    } else {
                        None // already cold
                    }
                })
                .collect()
        };

        if !has_targets {
            return;
        }

        // Hold reloading mutex across the cold-creation writes. This serializes
        // with ensure_columns/segments_snapshot so callers never see a cold
        // volume that ensure_columns' has_cold check just missed.
        let _reload_guard = self.reloading.lock();

        // Apply transitions under write lock. Arc<VolumeMetadata> is shared
        // (zero-copy), only LazyColumns is replaced.
        let mut segments = self.segments.write();
        let mut new_map = (**segments).clone();
        for &(seg_id, is_hot, is_warm) in &targets {
            if is_hot {
                // Hot → Warm: drop decompressed columns, keep compressed in RAM
                if let Some(cs) = new_map.get_mut(&seg_id) {
                    if let Some(warm) = cs.volume.to_warm() {
                        cs.volume = Arc::new(warm);
                    }
                }
            } else if is_warm {
                // Warm → Cold: drop compressed blocks, keep metadata in map.
                // Zone maps, stats, row_ids stay available for fast paths.
                // Only column access triggers disk reload via ensure_loaded.
                if let Some(cs) = new_map.get_mut(&seg_id) {
                    let cold = cs.volume.to_cold();
                    cs.volume = Arc::new(cold);
                    self.has_cold
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        *segments = Arc::new(new_map);
    }

    /// Reload cold volumes (metadata-only, in segments map) from disk.
    /// Replaces them in-place with full deferred volumes.
    fn reload_cold_volumes(&self, ids: Vec<u64>) {
        let vol_dir = match &self.volume_dir {
            Some(d) => d,
            None => return,
        };
        let mut reloaded = Vec::new();
        let mut failed = Vec::new();
        for &id in &ids {
            let filename = format!("vol_{:016x}.vol", id);
            let full_path = vol_dir.join(self.table_name.as_str()).join(filename);
            match crate::storage::volume::io::read_volume_from_disk(&full_path) {
                Ok(volume) => {
                    reloaded.push((id, Arc::new(volume)));
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to reload cold volume {} seg={}: {}",
                        self.table_name, id, e
                    );
                    failed.push(id);
                }
            }
        }
        if reloaded.is_empty() {
            // Nothing loaded (all failed or empty list). Leave cold volumes
            // in place — next access retries. Don't remove from manifest
            // (transient I/O errors shouldn't cause permanent data loss).
            return;
        }
        let mut segments = self.segments.write();
        let mut new_map = (**segments).clone();
        for (id, volume) in reloaded {
            if let Some(cs) = new_map.get_mut(&id) {
                if !cs.volume.unique_indices.read().is_empty() {
                    *volume.unique_indices.write() =
                        std::mem::take(&mut *cs.volume.unique_indices.write());
                }
                volume.mark_accessed();
                cs.volume = volume;
            }
        }
        let still_cold = new_map.values().any(|cs| cs.volume.is_cold());
        *segments = Arc::new(new_map);
        if !still_cold {
            self.has_cold
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Count segments below the target row count (sub-target volumes that need merging).
    pub fn sub_target_segment_count(&self, target_rows: usize) -> usize {
        let manifest = self.manifest.read();
        manifest
            .segments
            .iter()
            .filter(|s| s.row_count < target_rows)
            .count()
    }

    /// Check if a segment with the given ID is already registered (loaded in memory).
    pub fn has_segment(&self, segment_id: u64) -> bool {
        self.segments.read().contains_key(&segment_id)
    }

    /// Check if a segment exists in the manifest (source of truth for what should be loaded).
    /// Cheaper than `load_volume_for_existing_segment` — no volume data needed.
    pub fn manifest_has_segment(&self, segment_id: u64) -> bool {
        self.manifest
            .read()
            .segments
            .iter()
            .any(|s| s.segment_id == segment_id)
    }

    /// Register a new segment (after seal, compaction, or load).
    /// When `invalidate_cache` is false (compaction), the unique lookup cache is
    /// preserved. Compaction doesn't change row_ids or values, just which volume
    /// they live in. The cache's `row_exists()` filter handles stale entries.
    pub fn register_segment(
        &self,
        segment_id: u64,
        volume: Arc<FrozenVolume>,
        meta: SegmentMeta,
        schema: Option<&crate::core::Schema>,
    ) {
        self.register_segment_inner(segment_id, volume, meta, schema);
    }

    fn register_segment_inner(
        &self,
        segment_id: u64,
        volume: Arc<FrozenVolume>,
        meta: SegmentMeta,
        schema: Option<&crate::core::Schema>,
    ) {
        // Both manifest and segments must be updated atomically under write locks.
        // The bitmap computation runs inside the critical section — this is safe
        // because the segments write lock only blocks other writers (readers clone
        // the Arc), and concurrent writers are already serialized by seal_fence.
        {
            let seg_schema_version = meta.schema_version;
            let mut manifest = self.manifest.write();
            if segment_id >= manifest.next_segment_id {
                manifest.next_segment_id = segment_id + 1;
            }
            manifest.add_segment(meta);
            let mapping = if let Some(s) = schema {
                let drops = manifest.dropped_columns.clone();
                let renames = manifest.column_renames.clone();
                super::writer::compute_column_mapping_with_drops(
                    s,
                    &volume,
                    &drops,
                    seg_schema_version,
                    &renames,
                )
            } else {
                super::writer::ColumnMapping {
                    sources: (0..volume.columns.len())
                        .map(super::writer::ColSource::Volume)
                        .collect(),
                    is_identity: true,
                }
            };
            let cold = ColdSegment {
                volume,
                mapping,
                schema_version: seg_schema_version,
                visible: None,
            };
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            new_map.insert(segment_id, cold);
            compute_visibility_bitmaps(&seg_ids, &mut new_map, &mut self.visibility_seen.lock());
            *segments = Arc::new(new_map);
        }
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.has_segments_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.seal_generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    /// Load a volume into the segments map for an existing manifest entry.
    /// Returns true if the segment_id was found in the manifest and loaded,
    /// false if the segment_id is not in the manifest.
    /// This avoids adding duplicate segment metadata when the manifest was
    /// pre-loaded from disk and volumes are loaded separately.
    pub fn load_volume_for_existing_segment(
        &self,
        segment_id: u64,
        volume: Arc<FrozenVolume>,
    ) -> bool {
        let manifest = self.manifest.read();
        let seg_schema_version = manifest
            .segments
            .iter()
            .find(|s| s.segment_id == segment_id)
            .map(|s| s.schema_version);
        // Column renames are already merged into column_name_map before Arc
        // wrapping in load_standalone_volumes.
        drop(manifest);

        if let Some(schema_version) = seg_schema_version {
            let cold = ColdSegment {
                mapping: super::writer::ColumnMapping {
                    sources: (0..volume.columns.len())
                        .map(super::writer::ColSource::Volume)
                        .collect(),
                    is_identity: true,
                },
                volume,
                schema_version,
                visible: None,
            };
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            new_map.insert(segment_id, cold);
            *segments = Arc::new(new_map);
            self.has_segments_flag
                .store(true, std::sync::atomic::Ordering::Relaxed);

            true
        } else {
            false
        }
    }

    /// Rename this segment manager's table (for ALTER TABLE RENAME).
    pub fn rename(&mut self, new_name: &str) {
        self.table_name = SmartString::from(new_name);
        self.manifest.write().table_name = SmartString::from(new_name);
    }

    /// Add tombstone row_ids. `commit_seq` enables snapshot isolation
    /// (older snapshots only see tombstones with `commit_seq <= their begin_seq`).
    /// `visible_at_lsn` is the WAL LSN at which the tombstone first
    /// becomes cross-process visible — capped read-only attaches hide
    /// tombstones with `visible_at_lsn > attach_visible_commit_lsn` so
    /// they can't suppress a row that was visible at the cap. Pass
    /// the writer's `current_wal_lsn()` at tombstone-add time.
    ///
    /// Lock order: manifest FIRST, then tombstones (matches read paths like
    /// deduped_row_count, total_row_count, check_value_exists_in_segments).
    pub fn add_tombstones(&self, row_ids: &[i64], commit_seq: u64, visible_at_lsn: u64) {
        if row_ids.is_empty() {
            return;
        }
        let mut manifest = self.manifest.write();
        let mut ts_guard = self.tombstones.write();
        let ts = Arc::make_mut(&mut *ts_guard);
        let mut changed = false;
        for &rid in row_ids {
            use std::collections::hash_map::Entry;
            match ts.entry(rid) {
                Entry::Vacant(e) => {
                    e.insert(commit_seq);
                    manifest.tombstones.push((rid, commit_seq, visible_at_lsn));
                    changed = true;
                }
                Entry::Occupied(mut e) => {
                    // Update existing tombstone if the new commit_seq is
                    // different. This ensures repeated seal-skip tombstones
                    // get a fresh sequence that won't match an older
                    // compaction snapshot.
                    if *e.get() != commit_seq {
                        let old_seq = *e.get();
                        e.insert(commit_seq);
                        // Update the manifest entry in-place. visible_at_lsn
                        // also advances to the new value — re-recording a
                        // tombstone makes it cross-process visible at the
                        // new LSN, so capped readers below the new LSN
                        // continue to see the original cold row (correct).
                        if let Some(entry) = manifest
                            .tombstones
                            .iter_mut()
                            .find(|(r, s, _)| *r == rid && *s == old_seq)
                        {
                            entry.1 = commit_seq;
                            entry.2 = visible_at_lsn;
                        }
                        changed = true;
                    }
                }
            }
        }
        if changed {
            self.cached_deduped_count
                .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Clear all tombstones (after compaction has resolved them).
    /// Lock order: manifest FIRST, then tombstones.
    pub fn clear_tombstones(&self) {
        self.manifest.write().tombstones.clear();
        *self.tombstones.write() = Arc::new(FxHashMap::default());
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Remove tombstones only for row_ids that were in the compacted volumes.
    /// Used by partial compaction (merging a subset of volumes) where tombstones
    /// for unmerged volumes must remain.
    pub fn remove_tombstones_for_rows(&self, row_ids: &FxHashSet<i64>) {
        if row_ids.is_empty() {
            return;
        }
        let mut manifest = self.manifest.write();
        let mut ts_guard = self.tombstones.write();
        let ts = Arc::make_mut(&mut *ts_guard);
        let before = ts.len();
        ts.retain(|rid, _| !row_ids.contains(rid));
        if ts.len() != before {
            manifest
                .tombstones
                .retain(|&(rid, _, _)| !row_ids.contains(&rid));
            self.cached_deduped_count
                .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Remove only tombstones that match both row_id AND commit_seq from a
    /// prior snapshot. Tombstones added after the snapshot (e.g., by a
    /// concurrent seal) are preserved.
    pub fn remove_tombstones_matching_snapshot(
        &self,
        snapshot: &FxHashMap<i64, u64>,
        row_ids: &FxHashSet<i64>,
    ) {
        if row_ids.is_empty() || snapshot.is_empty() {
            return;
        }
        let mut manifest = self.manifest.write();
        let mut ts_guard = self.tombstones.write();
        let ts = Arc::make_mut(&mut *ts_guard);
        let before = ts.len();
        ts.retain(|rid, seq| {
            if !row_ids.contains(rid) {
                return true; // not in merged volumes, keep
            }
            // Only remove if the commit_seq matches the snapshot.
            // If a newer tombstone was added (different seq), keep it.
            !matches!(snapshot.get(rid), Some(snap_seq) if *snap_seq == *seq)
        });
        if ts.len() != before {
            manifest.tombstones.retain(|&(rid, seq, _)| {
                if !row_ids.contains(&rid) {
                    return true;
                }
                !matches!(snapshot.get(&rid), Some(snap_seq) if *snap_seq == seq)
            });
            self.cached_deduped_count
                .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Get an Arc reference to the tombstone map. O(1) — no data clone.
    /// Use this for read-only access (membership checks, iteration).
    /// Keys are row_ids, values are commit_seq (for snapshot filtering).
    pub fn tombstone_set_arc(&self) -> Arc<FxHashMap<i64, u64>> {
        Arc::clone(&*self.tombstones.read())
    }

    /// Check if the tombstone set is empty without cloning.
    pub fn is_tombstone_set_empty(&self) -> bool {
        self.tombstones.read().is_empty()
    }

    /// Get write access to the tombstone map (for seal cleanup).
    pub fn tombstones_write(&self) -> parking_lot::RwLockWriteGuard<'_, Arc<FxHashMap<i64, u64>>> {
        self.tombstones.write()
    }

    // ---- Per-transaction pending tombstones ----

    /// Track a cold row_id as pending tombstone for a transaction.
    /// Called during DML (UPDATE/DELETE of cold rows).
    pub fn add_pending_tombstone(&self, txn_id: i64, row_id: i64) {
        self.pending_txn_tombstones
            .write()
            .entry(txn_id)
            .or_default()
            .insert(row_id);
    }

    /// Get pending tombstone row_ids for a transaction (for WAL recording).
    pub fn get_pending_tombstones(&self, txn_id: i64) -> Vec<i64> {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Insert pending tombstones for a transaction directly into a set (no Vec clone).
    pub fn insert_pending_tombstones_into(
        &self,
        txn_id: i64,
        dest: &mut rustc_hash::FxHashSet<i64>,
    ) {
        if let Some(ids) = self.pending_txn_tombstones.read().get(&txn_id) {
            for &id in ids {
                dest.insert(id);
            }
        }
    }

    /// Get the count of pending tombstones for a transaction without cloning.
    pub fn pending_tombstone_count(&self, txn_id: i64) -> usize {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .map_or(0, |v| v.len())
    }

    /// Check if a specific row_id is a pending tombstone for a transaction.
    /// O(1) with FxHashSet (was O(n) with Vec).
    pub fn is_pending_tombstone(&self, txn_id: i64, row_id: i64) -> bool {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .is_some_and(|set| set.contains(&row_id))
    }

    /// Commit pending tombstones: move from per-txn pending to shared tombstone set.
    /// The commit_seq is the transaction's commit sequence, used for snapshot
    /// isolation: older snapshots won't see these tombstones.
    /// `visible_at_lsn` stamps the cross-process visibility frontier
    /// (typically the writer's `current_wal_lsn` at commit time).
    pub fn commit_pending_tombstones(&self, txn_id: i64, commit_seq: u64, visible_at_lsn: u64) {
        let pending = self.pending_txn_tombstones.write().remove(&txn_id);
        if let Some(ids) = pending {
            if !ids.is_empty() {
                let id_vec: Vec<i64> = ids.into_iter().collect();
                self.add_tombstones(&id_vec, commit_seq, visible_at_lsn);
            }
        }
    }

    /// Rollback pending tombstones: discard without applying.
    pub fn rollback_pending_tombstones(&self, txn_id: i64) {
        self.pending_txn_tombstones.write().remove(&txn_id);
    }

    /// Check if a txn has any pending tombstones (for has_local_changes).
    pub fn has_pending_tombstones(&self, txn_id: i64) -> bool {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .is_some_and(|v| !v.is_empty())
    }

    /// Check if a row_id is tombstoned (any commit_seq).
    pub fn is_tombstoned(&self, row_id: i64) -> bool {
        self.tombstones.read().contains_key(&row_id)
    }

    /// Check if a row_id exists in any segment (not tombstoned).
    ///
    /// Used for constraint checking (PK/UNIQUE).
    pub fn row_exists(&self, row_id: i64) -> bool {
        // Atomic snapshot: capture tombstones AND manifest segment
        // metadata AND segments map under one `manifest.read()` so a
        // concurrent read-only `reload_from_disk` can't swap any of
        // them between reads. Same race as `get_cold_row_normalized`.
        let (ts, seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .map(|m| (m.segment_id, m.min_row_id, m.max_row_id))
                .collect();
            let ts = Arc::clone(&*self.tombstones.read());
            let segments = Arc::clone(&*self.segments.read());
            (ts, seg_ids, segments)
        };
        if ts.contains_key(&row_id) {
            return false;
        }
        // Metadata-only check (binary search on row_ids). Does not access
        // column data, so no mark_accessed — should not pin volumes.
        for (seg_id, min_id, max_id) in &seg_ids {
            if row_id < *min_id || row_id > *max_id {
                continue;
            }
            if let Some(cold) = segments.get(seg_id) {
                if cold.volume.meta.row_ids.binary_search(&row_id).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Get a cold row by row_id. Returns the Row if found and not tombstoned.
    /// Iterates newest-first so overlapping row_ids return the newest version.
    /// Uses metadata-only search, reloads only the target cold volume if needed.
    pub fn get_cold_row(&self, row_id: i64) -> Option<crate::core::Row> {
        // Atomic snapshot: same lock discipline as
        // `get_cold_row_normalized` and `row_exists`.
        let (ts, seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| (m.segment_id, m.min_row_id, m.max_row_id))
                .collect();
            let ts = Arc::clone(&*self.tombstones.read());
            let segments = Arc::clone(&*self.segments.read());
            (ts, seg_ids, segments)
        };
        if ts.contains_key(&row_id) {
            return None;
        }
        for (seg_id, min_id, max_id) in &seg_ids {
            if row_id < *min_id || row_id > *max_id {
                continue;
            }
            if let Some(cold) = segments.get(seg_id) {
                if let Ok(idx) = cold.volume.meta.row_ids.binary_search(&row_id) {
                    if cold.volume.is_cold() {
                        if let Some(vol) = self.ensure_volume(*seg_id) {
                            return Some(vol.get_row(idx));
                        }
                        // Segment removed by compaction — retry with fresh state.
                        return self.get_cold_row_retry(row_id);
                    }
                    cold.volume.mark_accessed();
                    return Some(cold.volume.get_row(idx));
                }
            }
        }
        None
    }

    /// Retry get_cold_row with a fresh consistent snapshot after compaction.
    fn get_cold_row_retry(&self, row_id: i64) -> Option<crate::core::Row> {
        self.ensure_columns();
        // Re-check tombstones too: a refresh between the failed
        // `ensure_volume` and this retry can introduce a tombstone
        // that hides the row in its new compacted location.
        let (ts, seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| m.segment_id)
                .collect();
            let ts = Arc::clone(&*self.tombstones.read());
            let segments = Arc::clone(&*self.segments.read());
            (ts, seg_ids, segments)
        };
        if ts.contains_key(&row_id) {
            return None;
        }
        for seg_id in &seg_ids {
            if let Some(cold) = segments.get(seg_id) {
                if let Ok(idx) = cold.volume.meta.row_ids.binary_search(&row_id) {
                    cold.volume.mark_accessed();
                    return Some(cold.volume.get_row(idx));
                }
            }
        }
        None
    }

    /// Get a cold row by row_id, normalized to the current schema.
    /// After ALTER TABLE ADD COLUMN, cold volumes may have fewer columns.
    /// This variant fills in defaults for missing columns.
    /// Iterates newest-first so overlapping row_ids return the newest version.
    /// Uses metadata-only search, reloads only the target cold volume if needed.
    pub fn get_cold_row_normalized(
        &self,
        row_id: i64,
        schema: &crate::core::Schema,
    ) -> Option<crate::core::Row> {
        // Atomic snapshot: tombstones + manifest segment metadata +
        // segments map under a single `manifest.read()`. The previous
        // flow read tombstones first, then took a separate snapshot —
        // a concurrent read-only `reload_from_disk` (which acquires
        // manifest→tombstones→segments writes together) could swap
        // between the two reads and yield a mixed snapshot, returning
        // a row using old tombstones with new segments or hiding a
        // live row using new tombstones with old segments.
        let (ts, seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| (m.segment_id, m.min_row_id, m.max_row_id))
                .collect();
            let ts = Arc::clone(&*self.tombstones.read());
            let segments = Arc::clone(&*self.segments.read());
            (ts, seg_ids, segments)
        };
        if ts.contains_key(&row_id) {
            return None;
        }
        for (seg_id, min_id, max_id) in &seg_ids {
            if row_id < *min_id || row_id > *max_id {
                continue;
            }
            if let Some(cold) = segments.get(seg_id) {
                if let Ok(idx) = cold.volume.meta.row_ids.binary_search(&row_id) {
                    // Capture the mapping from the SAME ColdSegment
                    // we matched on. A live `get_volume_mapping`
                    // call would race a concurrent
                    // `reload_from_disk` that replaced or removed
                    // this segment between our snapshot and now,
                    // returning either a different segment's
                    // mapping or the empty identity fallback —
                    // which would materialize the row from the
                    // captured volume with the wrong column
                    // alignment.
                    let mapping = cold.mapping.clone();
                    let vol = if cold.volume.is_cold() {
                        match self.ensure_volume(*seg_id) {
                            Some(v) => v,
                            None => {
                                // Segment removed by compaction — retry with fresh state.
                                return self.get_cold_row_normalized_retry(row_id, schema);
                            }
                        }
                    } else {
                        cold.volume.mark_accessed();
                        Arc::clone(&cold.volume)
                    };
                    if mapping.is_identity {
                        return Some(vol.get_row(idx));
                    }
                    return Some(vol.get_row_mapped(idx, &mapping));
                }
            }
        }
        let _ = schema;
        None
    }

    /// Retry get_cold_row_normalized with a fresh consistent snapshot after compaction.
    fn get_cold_row_normalized_retry(
        &self,
        row_id: i64,
        schema: &crate::core::Schema,
    ) -> Option<crate::core::Row> {
        self.ensure_columns();
        // Atomic snapshot: same lock discipline as the primary path.
        // The retry must re-check tombstones too — a refresh between
        // the failed `ensure_volume` and this retry can introduce a
        // tombstone that hides the row in its new compacted location.
        let (ts, seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<u64> = manifest
                .segments
                .iter()
                .rev()
                .map(|m| m.segment_id)
                .collect();
            let ts = Arc::clone(&*self.tombstones.read());
            let segments = Arc::clone(&*self.segments.read());
            (ts, seg_ids, segments)
        };
        if ts.contains_key(&row_id) {
            return None;
        }
        for seg_id in &seg_ids {
            if let Some(cold) = segments.get(seg_id) {
                if let Ok(idx) = cold.volume.meta.row_ids.binary_search(&row_id) {
                    cold.volume.mark_accessed();
                    // Same race-free mapping capture as the
                    // primary path — see the comment in
                    // `get_cold_row_normalized` above.
                    let mapping = cold.mapping.clone();
                    if mapping.is_identity {
                        return Some(cold.volume.get_row(idx));
                    }
                    return Some(cold.volume.get_row_mapped(idx, &mapping));
                }
            }
        }
        let _ = schema;
        None
    }

    /// Check if a row_id actually exists in any loaded volume.
    /// Does NOT check tombstones. Used for idempotent WAL replay.
    /// Uses binary search on the volume's row_ids for O(log n) per segment.
    pub fn is_row_id_in_volume(&self, row_id: i64) -> bool {
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .map(|m| (m.segment_id, m.min_row_id, m.max_row_id))
                .collect();
            let segments = Arc::clone(&*self.segments.read());
            (seg_ids, segments)
        };
        for (seg_id, min_id, max_id) in &seg_ids {
            if row_id < *min_id || row_id > *max_id {
                continue;
            }
            if let Some(cold) = segments.get(seg_id) {
                if cold.volume.meta.row_ids.binary_search(&row_id).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Upper-bound estimate of the row count across all segments.
    /// Used by the `ReadTable::row_count_hint` contract for planner /
    /// cache decisions; it does NOT deduplicate overlapping row_ids
    /// across segments and does NOT subtract tombstones. Tombstones
    /// can outlive the segments they originally pointed at (compaction
    /// merges segments and the merged volume physically excludes
    /// tombstoned rows, but the manifest may still carry the tombstone
    /// entries for the old row_ids). Subtracting `tombstones.len()`
    /// blindly therefore over-subtracts and can drive the hint to
    /// zero on a large table whose only "tombstones" are those
    /// orphans, fooling the planner / cache into treating the table
    /// as small.
    ///
    /// Per the upper-bound contract, the only safe cheap thing is
    /// the sum of segment row counts. Use `deduped_row_count()` for
    /// an exact count.
    pub fn total_row_count(&self) -> usize {
        let manifest = self.manifest.read();
        manifest.segments.iter().map(|s| s.row_count).sum()
    }

    /// Get the exact deduplicated row count across all segments.
    /// Uses a cached value that is invalidated on segment/tombstone changes.
    pub fn deduped_row_count(&self) -> usize {
        let cached = self
            .cached_deduped_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if cached != u64::MAX {
            return cached as usize;
        }
        let count = self.compute_deduped_row_count();
        self.cached_deduped_count
            .store(count as u64, std::sync::atomic::Ordering::Relaxed);
        count
    }

    /// Acquire a shared guard while performing a cold check + hot insert.
    /// Seal takes the exclusive guard so it cannot move rows between the
    /// cold visibility check and hot publication.
    #[inline]
    pub fn acquire_seal_read(&self) -> parking_lot::RwLockReadGuard<'_, ()> {
        self.seal_fence.read()
    }

    /// Acquire the exclusive guard for the seal critical section.
    #[inline]
    pub fn acquire_seal_write(&self) -> parking_lot::RwLockWriteGuard<'_, ()> {
        self.seal_fence.write()
    }

    /// Current seal generation. Incremented on every register_segment.
    #[inline]
    pub fn seal_generation(&self) -> u64 {
        self.seal_generation
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Record the current seal generation for a transaction. Called under
    /// the seal read fence during INSERT so the value is consistent.
    /// Stores the minimum (earliest) generation seen by this txn, so that
    /// a later INSERT within the same txn cannot hide an earlier seal.
    #[inline]
    pub fn record_txn_seal_generation(&self, txn_id: i64) {
        let gen = self.seal_generation();
        let mut map = self.txn_seal_gens.lock();
        map.entry(txn_id)
            .and_modify(|existing| {
                if gen < *existing {
                    *existing = gen;
                }
            })
            .or_insert(gen);
    }

    /// Get the seal generation recorded for a transaction.
    #[inline]
    pub fn get_txn_seal_generation(&self, txn_id: i64) -> Option<u64> {
        self.txn_seal_gens.lock().get(&txn_id).copied()
    }

    /// Remove the seal generation record for a transaction (on commit/rollback).
    #[inline]
    pub fn clear_txn_seal_generation(&self, txn_id: i64) {
        self.txn_seal_gens.lock().remove(&txn_id);
    }

    /// Count visible rows using pre-computed visibility bitmaps.
    /// Falls back to hash-based dedup only when bitmaps are not available.
    fn compute_deduped_row_count(&self) -> usize {
        let segments = Arc::clone(&*self.segments.read());
        if segments.is_empty() {
            return 0;
        }
        let tombstones = Arc::clone(&*self.tombstones.read());

        // The previous single-segment fast path returned
        // `total - tombstones.len()`, which subtracted EVERY manifest
        // tombstone from the segment's row count. After compaction
        // collapses N segments into 1, the manifest may still carry
        // tombstones for row_ids that the merged volume already
        // physically excluded; subtracting them under-counts the
        // current segment by the number of those orphan tombstones.
        // SWMR readers also see this whenever a writer compacts down
        // to a single segment between checkpoints, producing the
        // "COUNT(*) wrong on read-only" bug. The per-row path below
        // is correct for any segment count, including 1, because
        // `tombstones.contains_key(rid)` only filters row_ids that
        // are actually present in this segment.
        let mut count = 0usize;
        for cs in segments.values() {
            let vol = &cs.volume;
            for i in 0..vol.meta.row_count {
                if !cs.is_visible(i) {
                    continue;
                }
                if !tombstones.is_empty() && tombstones.contains_key(&vol.meta.row_ids[i]) {
                    continue;
                }
                count += 1;
            }
        }
        count
    }

    /// Get the cached column mapping for a volume. Computes on first call,
    /// returns cached on subsequent calls. Handles dropped columns + renames
    /// automatically. Call invalidate_mappings() on ALTER TABLE.
    pub fn get_volume_mapping(
        &self,
        seg_id: u64,
        _schema: &crate::core::Schema,
    ) -> super::writer::ColumnMapping {
        let segs = self.segments.read();
        if let Some(cold) = segs.get(&seg_id) {
            cold.mapping.clone()
        } else {
            super::writer::ColumnMapping {
                sources: Vec::new(),
                is_identity: true,
            }
        }
    }

    /// Recompute all column mappings for loaded volumes.
    /// Called on ALTER TABLE (rename/drop/add column).
    pub fn invalidate_mappings(&self, schema: &crate::core::Schema) {
        let manifest = self.manifest.read();
        let drops = manifest.dropped_columns.clone();
        let renames = manifest.column_renames.clone();
        drop(manifest);
        let mut segs = self.segments.write();
        let mut new_map = (**segs).clone();
        for cold in new_map.values_mut() {
            cold.mapping = super::writer::compute_column_mapping_with_drops(
                schema,
                &cold.volume,
                &drops,
                cold.schema_version,
                &renames,
            );
        }
        *segs = Arc::new(new_map);
    }

    /// Bump the manifest's table-level `schema_version` to at
    /// least `version`. Called by the engine after every DDL
    /// operation that affects this table — including ADD/MODIFY
    /// COLUMN that don't otherwise touch the manifest. The
    /// monotonic max ensures concurrent / out-of-order writes
    /// from different DDL paths can't move the version
    /// backwards. Persisted on the next manifest write so a
    /// no-shm reader's drift check sees it.
    pub fn record_table_schema_version(&self, version: u64) {
        let mut manifest = self.manifest.write();
        if version > manifest.schema_version {
            manifest.schema_version = version;
        }
    }

    /// Record a column drop so old volumes don't leak stale data.
    /// `schema_version` is the current schema epoch at drop time. Only volumes
    /// with schema_version <= this value will have the column masked.
    pub fn record_column_drop(&self, col_name: &str, schema_version: u64) {
        let lower = SmartString::from(col_name.to_lowercase());
        let mut manifest = self.manifest.write();
        // Remove any existing entry for this column name before adding the new one.
        // This handles DROP + ADD + DROP sequences correctly.
        manifest
            .dropped_columns
            .retain(|(name, _)| name.as_str() != lower.as_str());
        manifest.dropped_columns.push((lower, schema_version));
    }

    /// Note: record_column_readd was removed. dropped_columns is permanent
    /// until compaction rewrites all old volumes. After ADD COLUMN re-adds a
    /// dropped name, compute_column_mapping_with_drops handles it correctly:
    /// old volumes have the column at an old position (blocked by drop mask),
    /// new volumes don't have it at all (mapped to Default).
    ///
    /// Clear dropped column tracking (called after compaction replaces all old volumes).
    pub fn clear_dropped_columns(&self) {
        self.manifest.write().dropped_columns.clear();
    }

    /// Check if a column name was dropped (for compute_column_mapping).
    pub fn is_column_dropped(&self, col_name: &str) -> bool {
        self.manifest
            .read()
            .dropped_columns
            .iter()
            .any(|(name, _)| name.as_str() == col_name)
    }

    pub fn get_dropped_columns(&self) -> Vec<(SmartString, u64)> {
        self.manifest.read().dropped_columns.clone()
    }

    pub fn set_seal_overlap(&self, count: usize) {
        self.seal_overlap_count
            .store(count, std::sync::atomic::Ordering::Release);
    }

    /// Clear the seal overlap count. Called AFTER remove_sealed_rows completes.
    pub fn clear_seal_overlap(&self) {
        self.seal_overlap_count
            .store(0, std::sync::atomic::Ordering::Release);
    }

    /// Get the current seal overlap count (for row_count correction).
    pub fn seal_overlap(&self) -> usize {
        self.seal_overlap_count
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// Persist the manifest to disk (includes tombstones).
    pub fn persist(&self) -> Result<()> {
        self.persist_manifest_only()
    }

    /// Persist only the manifest.
    pub fn persist_manifest_only(&self) -> Result<()> {
        let Some(ref vol_dir) = self.volume_dir else {
            return Ok(());
        };

        let persist_name = self.manifest.read().table_name.clone();
        let table_dir = vol_dir.join(persist_name.as_str());
        std::fs::create_dir_all(&table_dir).map_err(|e| {
            crate::core::Error::internal(format!("failed to create table dir: {}", e))
        })?;

        let manifest_path = table_dir.join("manifest.bin");
        self.manifest.read().write_to_disk(&manifest_path)?;

        Ok(())
    }

    /// Record a column rename. The caller must call invalidate_mappings()
    /// afterwards to recompute column mappings with the new rename.
    pub fn record_column_rename(&self, old_name: &str, new_name: &str) {
        // Persist in manifest for restart
        self.manifest
            .write()
            .column_renames
            .push((SmartString::from(old_name), SmartString::from(new_name)));
    }

    /// Load manifest from disk.
    pub fn load_from_disk(table_name: &str, volume_dir: &Path) -> Result<Option<Self>> {
        let table_dir = volume_dir.join(table_name);
        let manifest_path = table_dir.join("manifest.bin");

        if !manifest_path.exists() {
            return Ok(None);
        }

        let manifest = TableManifest::read_from_disk(&manifest_path)?;
        let manager = Self::from_manifest(manifest, Some(volume_dir.to_path_buf()));

        Ok(Some(manager))
    }

    /// Re-read `manifest.bin` from disk and reconcile the in-memory
    /// state with it. Used by cross-process readers (v1 SWMR) to pick up
    /// changes the writer has published since this manager was last
    /// loaded.
    ///
    /// `max_known_schema_version` is the engine's current `schema_epoch`
    /// — the highest schema version this process has WAL-replayed. If
    /// the new manifest contains any segment with `schema_version >
    /// max_known_schema_version`, the writer has done DDL events the
    /// reader hasn't seen and the segment cannot be safely decoded
    /// against the reader's stale schema. We surface
    /// `Error::SchemaChanged` and DO NOT mutate state — the reader's
    /// pre-call snapshot remains intact, and the caller can propagate
    /// the error so the user knows to reopen the `Database`.
    ///
    /// Reconciliation (when no schema drift is detected):
    /// - Replace `self.manifest` with the freshly-loaded one.
    /// - Replace `self.tombstones` from the new manifest's tombstone vec.
    /// - For each segment_id in the new manifest that is NOT already in
    ///   `self.segments`: load the .vol file from disk via
    ///   `load_volume_for_existing_segment` so subsequent scans see it.
    /// - For each segment_id in `self.segments` that is NOT in the new
    ///   manifest (the writer compacted it away): remove from segments.
    /// - Invalidate the cached dedup count so the next scan recomputes.
    ///
    /// Returns `true` if any segment was added or removed (caller can
    /// log / invalidate downstream caches if useful), `false` if the
    /// reload was a no-op.
    ///
    /// Concurrent in-flight queries that hold an `Arc<HashMap>` from
    /// `segments_snapshot()` continue using their old snapshot until
    /// they finish, then drop their reference. New queries see the
    /// reconciled state.
    /// schema-drift pre-check that does NOT mutate
    /// any in-memory state. Reads `manifest.bin` and returns
    /// `Err(SchemaChanged)` if any segment in the new manifest has
    /// `schema_version > max_known_schema_version`. Returns
    /// `Ok(false)` for no-op (memory-only mgr or missing manifest)
    /// and `Ok(true)` if a manifest was found and passes the check.
    ///
    /// Used by `MVCCEngine::reload_manifests` to verify EVERY
    /// per-table manifest before mutating any of them, so a later
    /// table's schema drift can't leave earlier tables half-applied.
    pub fn peek_schema_drift(&self, max_known_schema_version: u64) -> Result<bool> {
        let Some(new_manifest) = self.read_manifest_from_disk()? else {
            return Ok(false);
        };
        self.validate_manifest_for_reload(&new_manifest, max_known_schema_version)?;
        Ok(true)
    }

    /// Validate a staged manifest read before read-only SWMR reload. This
    /// performs every check that can fail because the writer's schema is newer
    /// than this reader's in-memory catalog, without mutating manager state.
    pub fn validate_manifest_for_reload(
        &self,
        new_manifest: &TableManifest,
        max_known_schema_version: u64,
    ) -> Result<()> {
        if let Some(seg) = new_manifest
            .segments
            .iter()
            .find(|s| s.schema_version > max_known_schema_version)
        {
            return Err(crate::core::Error::SchemaChanged(format!(
                "table '{}': segment schema_version={} > reader's schema_epoch={} \
                 (writer has done DDL since this handle opened); reopen the \
                 Database / ReadOnlyDatabase to pick up the new schema",
                self.table_name, seg.schema_version, max_known_schema_version
            )));
        }
        // Table-level schema epoch check: catches ALTER TABLE
        // ADD/MODIFY COLUMN that didn't produce a new segment
        // (and so the per-segment `schema_version` check above
        // doesn't fire) and didn't touch dropped_columns or
        // column_renames (so the metadata check below doesn't
        // fire either). The writer stamps the manifest's
        // `schema_version` with its `schema_epoch` on every DDL.
        // A no-shm reader has no WAL-tail DDL detection, so the
        // table-level epoch on disk is the only signal that the
        // writer has done DDL since this handle opened.
        if new_manifest.schema_version > max_known_schema_version {
            return Err(crate::core::Error::SchemaChanged(format!(
                "table '{}': manifest schema_version={} > reader's \
                 schema_epoch={} (writer has done DDL since this handle \
                 opened); reopen the Database / ReadOnlyDatabase to pick \
                 up the new schema",
                self.table_name, new_manifest.schema_version, max_known_schema_version
            )));
        }
        // Metadata-only ALTER drift: ALTER TABLE DROP/RENAME COLUMN
        // followed by a checkpoint that produces no new segment
        // bumps `dropped_columns`/`column_renames` on the manifest
        // without bumping any segment's `schema_version`. The
        // segment-version check above misses this case, so a no-shm
        // reader (no WAL-tail DDL detection) would silently keep
        // querying through a stale column layout. Compare the
        // metadata vectors against the in-memory manifest; any
        // drift in EITHER direction (added entries from a new DDL,
        // OR cleared entries from a compaction we haven't picked up
        // yet) means the writer has touched DDL and the reader's
        // cached schema is suspect.
        if self.detect_metadata_drift(new_manifest) {
            return Err(crate::core::Error::SchemaChanged(format!(
                "table '{}': manifest column metadata changed (writer has \
                 dropped or renamed columns since this handle opened); \
                 reopen the Database / ReadOnlyDatabase to pick up the \
                 new schema",
                self.table_name
            )));
        }
        Ok(())
    }

    /// Read only the table manifest's checkpoint frontier without mutating
    /// this manager. Used by read-only SWMR refresh to reject a cross-table
    /// manifest snapshot while the writer is between per-table manifest
    /// renames for the next checkpoint.
    pub fn peek_checkpoint_lsn(&self) -> Result<Option<u64>> {
        Ok(self.read_manifest_from_disk()?.map(|m| m.checkpoint_lsn))
    }

    /// Read this table's `manifest.bin` without mutating the manager.
    pub fn read_manifest_from_disk(&self) -> Result<Option<TableManifest>> {
        let vol_dir = match &self.volume_dir {
            Some(d) => d.clone(),
            None => return Ok(None),
        };
        let table_dir = vol_dir.join(self.table_name.as_str());
        let manifest_path = table_dir.join("manifest.bin");
        if !manifest_path.exists() {
            return Ok(None);
        }
        Ok(Some(TableManifest::read_from_disk(&manifest_path)?))
    }

    /// True when the on-disk manifest's column-DDL metadata
    /// (dropped / renamed columns) differs from this segment
    /// manager's in-memory copy. Used by the schema-drift checks
    /// in [`Self::peek_schema_drift`] and [`Self::reload_from_disk`]
    /// to catch ALTER TABLE DROP/RENAME COLUMN that did not also
    /// produce a new segment with a bumped per-segment
    /// `schema_version` — the segment-version drift check
    /// otherwise misses it. ADD/MODIFY COLUMN are caught by the
    /// table-level `schema_version` check below, not here:
    /// they don't touch dropped_columns or column_renames.
    fn detect_metadata_drift(&self, new_manifest: &TableManifest) -> bool {
        let cur = self.manifest.read();
        cur.dropped_columns != new_manifest.dropped_columns
            || cur.column_renames != new_manifest.column_renames
    }

    pub fn reload_from_disk(&self, max_known_schema_version: u64) -> Result<bool> {
        let Some(new_manifest) = self.read_manifest_from_disk()? else {
            return Ok(false);
        };
        self.reload_from_manifest(new_manifest, max_known_schema_version)
    }

    /// Reconcile this manager from a manifest that the caller already staged
    /// from disk. Using staged manifests lets `MVCCEngine::reload_manifests`
    /// validate a cross-table checkpoint group and then apply the exact same
    /// manifests, without re-reading a table after the writer has advanced it.
    pub fn reload_from_manifest(
        &self,
        new_manifest: TableManifest,
        max_known_schema_version: u64,
    ) -> Result<bool> {
        let vol_dir = match &self.volume_dir {
            Some(d) => d.clone(),
            None => return Ok(false), // memory-only manager
        };
        let table_dir = vol_dir.join(self.table_name.as_str());
        self.validate_manifest_for_reload(&new_manifest, max_known_schema_version)?;

        // Compute the new segment_id set BEFORE swapping anything in,
        // so we can diff against `self.segments` deterministically.
        let new_seg_ids: rustc_hash::FxHashSet<u64> =
            new_manifest.segments.iter().map(|m| m.segment_id).collect();
        let new_tombstone_map: FxHashMap<i64, u64> = new_manifest
            .tombstones
            .iter()
            .map(|&(rid, seq, _vis)| (rid, seq))
            .collect();

        // Load all NEW volume files BEFORE swapping
        // the manifest. The previous flow swapped manifest +
        // tombstones first, then silently skipped volumes that
        // failed to load (ENOENT race) — which left the published
        // manifest referencing segments absent from `self.segments`,
        // so scans would miss them and the next epoch bump would
        // mark the state "consumed" without ever retrying. The
        // race-skip rationale was sound for v1 (writer defers
        // unlink while readers are live) but the failure mode was
        // silent.
        //
        // New flow: pre-load all NEW segments. If ANY fails (race
        // or otherwise), return error WITHOUT mutating any state —
        // the reader's pre-call view stays consistent, and
        // `refresh()` won't advance the cached epoch, so the next
        // refresh will retry the whole reconcile.
        let to_load: Vec<u64> = {
            let segs = self.segments.read();
            new_seg_ids
                .iter()
                .copied()
                .filter(|id| !segs.contains_key(id))
                .collect()
        };
        let mut staged_loads: Vec<(u64, Arc<crate::storage::volume::writer::FrozenVolume>)> =
            Vec::with_capacity(to_load.len());
        for seg_id in &to_load {
            let filename = format!("vol_{:016x}.vol", seg_id);
            let full_path = table_dir.join(filename);
            match crate::storage::volume::io::read_volume_from_disk(&full_path) {
                Ok(v) => staged_loads.push((*seg_id, Arc::new(v))),
                Err(e) => {
                    return Err(crate::core::Error::internal(format!(
                        "table '{}': failed to load segment {} (vol_{:016x}.vol): {}; \
                         declining to publish partial manifest, retry refresh",
                        self.table_name, seg_id, seg_id, e
                    )));
                }
            }
        }

        // Build the COMPLETE replacement segment map BEFORE taking
        // any write lock. The previous flow swapped the manifest, then
        // removed gone segments under one write lock, then inserted
        // new segments under another (one per call to
        // `load_volume_for_existing_segment`). A scan that took
        // `manifest.read()` and `segments.read()` separately could
        // observe the new manifest pointing at segment_ids absent
        // from `segments` (between the manifest swap and the inserts)
        // OR still-present old segment_ids that the new manifest had
        // dropped — silently missing or double-counting cold rows.
        //
        // New flow: build the full new segments map up front (cheap
        // — no I/O, just Arc clones for kept segments + ColdSegment
        // construction for staged volumes), then publish manifest +
        // tombstones + segments together under ALL THREE write locks.
        // Readers see one consistent state transition.
        let staged_map: rustc_hash::FxHashMap<
            u64,
            Arc<crate::storage::volume::writer::FrozenVolume>,
        > = staged_loads.into_iter().collect();
        let mut new_segments_map: FxHashMap<u64, ColdSegment> = {
            let segs = self.segments.read();
            let mut new_map: FxHashMap<u64, ColdSegment> = FxHashMap::with_capacity_and_hasher(
                new_manifest.segments.len(),
                Default::default(),
            );
            for seg_meta in &new_manifest.segments {
                let seg_id = seg_meta.segment_id;
                if let Some(existing) = segs.get(&seg_id) {
                    new_map.insert(seg_id, existing.clone());
                } else if let Some(volume) = staged_map.get(&seg_id) {
                    let cold = ColdSegment {
                        mapping: super::writer::ColumnMapping {
                            sources: (0..volume.columns.len())
                                .map(super::writer::ColSource::Volume)
                                .collect(),
                            is_identity: true,
                        },
                        volume: Arc::clone(volume),
                        schema_version: seg_meta.schema_version,
                        visible: None,
                    };
                    new_map.insert(seg_id, cold);
                } else {
                    return Err(crate::core::Error::internal(format!(
                        "table '{}': segment {} in new manifest is neither in \
                         current segments nor in staged loads (internal \
                         invariant violated)",
                        self.table_name, seg_id
                    )));
                }
            }
            new_map
        };

        // Recompute per-segment visibility bitmaps for the new map.
        //
        // Without this step, kept segments retain `visible` bitmaps
        // computed against the PRIOR segment set, and newly-loaded
        // segments get `visible: None` (= all rows visible). Both are
        // wrong relative to the new set: a row_id that an old segment
        // contained may now be superseded by a newer segment the
        // reader just learned about, but the old segment's bitmap
        // still marks that row visible. The MergingScanner would then
        // return the same row_id from BOTH segments, producing
        // duplicate rows in cross-process SWMR readers when a writer
        // UPSERT lands a new sealed segment between two reader
        // refreshes.
        //
        // Visibility is purely a function of the current segment set
        // (newest-first dedup by row_id), so recomputing across
        // reload_from_manifest is exactly what writer-side seal /
        // compaction paths already do via remove_segments_inner /
        // replace_segments_atomic.
        let seg_ids: Vec<u64> = new_manifest.segments.iter().map(|m| m.segment_id).collect();
        compute_visibility_bitmaps(
            &seg_ids,
            &mut new_segments_map,
            &mut self.visibility_seen.lock(),
        );

        // Detect change for the return value BEFORE the swap — compare
        // the new segment-id set to the old one. We can't compare
        // ColdSegment Arcs because kept segments are Arc-cloned out of
        // the existing map, so a no-op manifest reload still produces
        // a fresh map Arc.
        let changed = {
            let segs = self.segments.read();
            new_seg_ids.len() != segs.len() || new_seg_ids.iter().any(|id| !segs.contains_key(id))
        };

        // Atomic publish: hold all three write locks across the swap so
        // any scan observes either the FULL old state or the FULL new
        // state, never a mix. Lock order matches the existing
        // manifest→tombstones convention used elsewhere
        // (`add_tombstones`, `clear_tombstones`); segments comes last
        // since no read path takes it before manifest.
        let has_any_segments = !new_segments_map.is_empty();
        {
            let mut manifest_guard = self.manifest.write();
            let mut tombstones_guard = self.tombstones.write();
            let mut segments_guard = self.segments.write();
            *manifest_guard = new_manifest;
            *tombstones_guard = Arc::new(new_tombstone_map);
            *segments_guard = Arc::new(new_segments_map);
        }

        self.has_segments_flag
            .store(has_any_segments, std::sync::atomic::Ordering::Relaxed);

        // Recompute dedup cache lazily next time a scan needs it.
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);

        Ok(changed)
    }

    /// Remove all segments and tombstones (for DROP TABLE / TRUNCATE).
    pub fn clear(&self) {
        {
            let mut manifest = self.manifest.write();
            manifest.segments.clear();
            manifest.tombstones.clear();
        }
        *self.segments.write() = Arc::new(FxHashMap::default());
        *self.tombstones.write() = Arc::new(FxHashMap::default());
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.has_segments_flag
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.has_cold
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Remove specific segments after compaction.
    pub fn remove_segments(&self, segment_ids: &[u64]) {
        self.remove_segments_inner(segment_ids, true);
    }

    /// Remove segments without invalidating the unique lookup cache (for compaction).
    pub fn remove_segments_compacted(&self, segment_ids: &[u64]) {
        self.remove_segments_inner(segment_ids, false);
    }

    fn remove_segments_inner(&self, segment_ids: &[u64], _invalidate_cache: bool) {
        let seg_ids: Vec<u64> = {
            let mut manifest = self.manifest.write();
            manifest.remove_segments(segment_ids);
            manifest.segments.iter().map(|m| m.segment_id).collect()
        };
        {
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            for &id in segment_ids {
                new_map.remove(&id);
            }
            compute_visibility_bitmaps(&seg_ids, &mut new_map, &mut self.visibility_seen.lock());
            if self.has_cold.load(std::sync::atomic::Ordering::Relaxed)
                && !new_map.values().any(|cs| cs.volume.is_cold())
            {
                self.has_cold
                    .store(false, std::sync::atomic::Ordering::Relaxed);
            }
            *segments = Arc::new(new_map);
        }
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Atomically replace old segments with a new compacted segment.
    /// Both operations happen under a single manifest+segments write lock,
    /// so concurrent queries never see the intermediate state where both
    /// old and new segments are registered (which causes duplicate scanning).
    pub fn replace_segments_atomic(
        &self,
        new_segment_id: u64,
        new_volume: Arc<FrozenVolume>,
        new_meta: SegmentMeta,
        old_segment_ids: &[u64],
    ) {
        // Atomic: manifest + segments updated under both write locks.
        // Bitmap computation runs inside — safe because writers are serialized.
        {
            let mut manifest = self.manifest.write();
            let insert_pos = manifest
                .segments
                .iter()
                .position(|s| old_segment_ids.contains(&s.segment_id))
                .unwrap_or(manifest.segments.len());
            manifest.remove_segments(old_segment_ids);
            if new_segment_id >= manifest.next_segment_id {
                manifest.next_segment_id = new_segment_id + 1;
            }
            let insert_pos = insert_pos.min(manifest.segments.len());
            let seg_schema_version = new_meta.schema_version;
            manifest.segments.insert(insert_pos, new_meta);

            let cold = ColdSegment {
                mapping: super::writer::ColumnMapping {
                    sources: (0..new_volume.columns.len())
                        .map(super::writer::ColSource::Volume)
                        .collect(),
                    is_identity: true,
                },
                volume: new_volume,
                schema_version: seg_schema_version,
                visible: None,
            };
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            for &id in old_segment_ids {
                new_map.remove(&id);
            }
            new_map.insert(new_segment_id, cold);
            compute_visibility_bitmaps(&seg_ids, &mut new_map, &mut self.visibility_seen.lock());
            if self.has_cold.load(std::sync::atomic::Ordering::Relaxed)
                && !new_map.values().any(|cs| cs.volume.is_cold())
            {
                self.has_cold
                    .store(false, std::sync::atomic::Ordering::Relaxed);
            }
            *segments = Arc::new(new_map);
        }
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Atomically replace old segments with multiple new ones.
    /// Used by compaction-with-split when the merged output exceeds target_volume_rows.
    pub fn replace_segments_atomic_multi(
        &self,
        new_volumes: Vec<(u64, Arc<FrozenVolume>, SegmentMeta)>,
        old_segment_ids: &[u64],
    ) {
        if new_volumes.is_empty() {
            self.replace_segments_atomic_remove_only(old_segment_ids);
            return;
        }
        if new_volumes.len() == 1 {
            let (id, vol, meta) = new_volumes.into_iter().next().unwrap();
            self.replace_segments_atomic(id, vol, meta, old_segment_ids);
            return;
        }
        {
            let mut manifest = self.manifest.write();
            let insert_pos = manifest
                .segments
                .iter()
                .position(|s| old_segment_ids.contains(&s.segment_id))
                .unwrap_or(manifest.segments.len());
            manifest.remove_segments(old_segment_ids);

            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            for &id in old_segment_ids {
                new_map.remove(&id);
            }

            let insert_pos = insert_pos.min(manifest.segments.len());
            for (i, (seg_id, vol, meta)) in new_volumes.into_iter().enumerate() {
                if seg_id >= manifest.next_segment_id {
                    manifest.next_segment_id = seg_id + 1;
                }
                let seg_schema_version = meta.schema_version;
                manifest.segments.insert(insert_pos + i, meta);

                let cold = ColdSegment {
                    mapping: super::writer::ColumnMapping {
                        sources: (0..vol.columns.len())
                            .map(super::writer::ColSource::Volume)
                            .collect(),
                        is_identity: true,
                    },
                    volume: vol,
                    schema_version: seg_schema_version,
                    visible: None,
                };
                new_map.insert(seg_id, cold);
            }

            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            compute_visibility_bitmaps(&seg_ids, &mut new_map, &mut self.visibility_seen.lock());
            if self.has_cold.load(std::sync::atomic::Ordering::Relaxed)
                && !new_map.values().any(|cs| cs.volume.is_cold())
            {
                self.has_cold
                    .store(false, std::sync::atomic::Ordering::Relaxed);
            }
            *segments = Arc::new(new_map);
        }
        self.has_segments_flag
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Atomically remove old segments without adding a replacement.
    /// Used when partial compaction finds all rows in merged volumes are tombstoned.
    pub fn replace_segments_atomic_remove_only(&self, old_segment_ids: &[u64]) {
        {
            let mut manifest = self.manifest.write();
            manifest.remove_segments(old_segment_ids);
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let mut segments = self.segments.write();
            let mut new_map = (**segments).clone();
            for &id in old_segment_ids {
                new_map.remove(&id);
            }
            let has_any = !new_map.is_empty();
            compute_visibility_bitmaps(&seg_ids, &mut new_map, &mut self.visibility_seen.lock());
            if self.has_cold.load(std::sync::atomic::Ordering::Relaxed)
                && !new_map.values().any(|cs| cs.volume.is_cold())
            {
                self.has_cold
                    .store(false, std::sync::atomic::Ordering::Relaxed);
            }
            *segments = Arc::new(new_map);
            self.has_segments_flag
                .store(has_any, std::sync::atomic::Ordering::Relaxed);
        }
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the manifest for reading (e.g., to iterate segment metadata).
    pub fn manifest(&self) -> parking_lot::RwLockReadGuard<'_, TableManifest> {
        self.manifest.read()
    }

    /// CoW snapshot of the loaded segments map.
    /// Reloads cold volumes first so column data is available.
    /// Does NOT mark volumes as accessed — callers that need eviction
    /// protection should mark the specific volumes they use.
    pub fn segments_snapshot(&self) -> Arc<FxHashMap<u64, ColdSegment>> {
        self.ensure_columns();
        let segs = Arc::clone(&*self.segments.read());
        if !segs.values().any(|cs| cs.volume.is_cold()) {
            return segs;
        }
        // Race or persistent failure: retry once then filter.
        self.ensure_columns();
        let segs = Arc::clone(&*self.segments.read());
        if segs.values().any(|cs| cs.volume.is_cold()) {
            let mut filtered = (*segs).clone();
            let cold: Vec<(u64, usize)> = filtered
                .iter()
                .filter(|(_, cs)| cs.volume.is_cold())
                .map(|(&id, cs)| (id, cs.volume.meta.row_count))
                .collect();
            for &(seg_id, rows) in &cold {
                eprintln!(
                    "Warning: table {} seg={}: cold volume excluded from snapshot ({} rows, reload failed)",
                    self.table_name, seg_id, rows
                );
            }
            filtered.retain(|_, cs| !cs.volume.is_cold());
            return Arc::new(filtered);
        }
        segs
    }

    /// Raw CoW snapshot without ensure_columns. Callers that only need
    /// metadata (row_ids, zone maps) use this to avoid reloading cold volumes.
    pub fn segments_raw(&self) -> Arc<FxHashMap<u64, ColdSegment>> {
        Arc::clone(&*self.segments.read())
    }

    /// Reload a single cold volume by segment ID. Returns the loaded volume
    /// if successful. Used by point lookups to avoid reloading all cold volumes.
    pub fn ensure_volume(&self, seg_id: u64) -> Option<Arc<FrozenVolume>> {
        // Fast path: already loaded (or another thread just loaded it)
        {
            let segs = self.segments.read();
            if let Some(cs) = segs.get(&seg_id) {
                if !cs.volume.is_cold() {
                    cs.volume.mark_accessed();
                    return Some(Arc::clone(&cs.volume));
                }
            } else {
                // Segment no longer in map (compaction removed it). Caller
                // should retry with the current manifest.
                return None;
            }
        }
        // Serialize reloads — prevents concurrent stampede on the same volume.
        // Second thread re-checks the fast path after acquiring the guard.
        let _guard = self.reloading.lock();
        {
            let segs = self.segments.read();
            if let Some(cs) = segs.get(&seg_id) {
                if !cs.volume.is_cold() {
                    cs.volume.mark_accessed();
                    return Some(Arc::clone(&cs.volume));
                }
            } else {
                return None;
            }
        }
        let vol_dir = match &self.volume_dir {
            Some(d) => d,
            None => return None,
        };
        let filename = format!("vol_{:016x}.vol", seg_id);
        let full_path = vol_dir.join(self.table_name.as_str()).join(filename);
        let volume = match crate::storage::volume::io::read_volume_from_disk(&full_path) {
            Ok(v) => Arc::new(v),
            Err(e) => {
                eprintln!(
                    "Warning: Failed to reload cold volume {} seg={}: {}",
                    self.table_name, seg_id, e
                );
                return None;
            }
        };
        volume.mark_accessed();
        let mut segments = self.segments.write();
        let mut new_map = (**segments).clone();
        // Re-check: segment may have been removed by concurrent compaction
        // while we were reading from disk.
        if let Some(cs) = new_map.get_mut(&seg_id) {
            if !cs.volume.unique_indices.read().is_empty() {
                *volume.unique_indices.write() =
                    std::mem::take(&mut *cs.volume.unique_indices.write());
            }
            cs.volume = Arc::clone(&volume);
        } else {
            // Compaction removed this segment while we were reloading.
            // The row now lives in a newer compacted volume.
            return None;
        }
        let still_cold = new_map.values().any(|cs| cs.volume.is_cold());
        *segments = Arc::new(new_map);
        if !still_cold {
            self.has_cold
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        Some(volume)
    }

    /// Get the manifest for writing (e.g., to allocate segment IDs).
    pub fn manifest_mut(&self) -> parking_lot::RwLockWriteGuard<'_, TableManifest> {
        self.manifest.write()
    }

    /// SWMR v2 P1: drop segments AND tombstones whose
    /// `visible_at_lsn > cap_lsn` from this manager's manifest,
    /// segments map, and runtime tombstone map. Used by capped
    /// read-only attach so the engine never exposes cold rows from
    /// segments the writer published AFTER our
    /// `attach_visible_commit_lsn`, AND never hides a cold row via
    /// a tombstone the writer added after our cap.
    ///
    /// Without segment filtering: a writer checkpoint between the
    /// shm sample and manifest load would expose post-attach cold
    /// segments. Without tombstone filtering: a writer DELETE on a
    /// retained pre-cap segment after our cap would hide a row that
    /// was visible at our cap.
    ///
    /// `cap_lsn = u64::MAX` is a no-op (writable open / uncapped).
    /// V6 manifests have tombstone `visible_at_lsn = 0` (synthesized
    /// "always visible") and are unaffected by the tombstone filter.
    pub fn retain_segments_visible_at_or_below(&self, cap_lsn: u64) {
        if cap_lsn == u64::MAX {
            return;
        }
        // ---- Filter segments ----
        let removed_seg_ids: Vec<u64> = {
            let mut manifest = self.manifest.write();
            let removed: Vec<u64> = manifest
                .segments
                .iter()
                .filter(|s| s.visible_at_lsn > cap_lsn)
                .map(|s| s.segment_id)
                .collect();
            if !removed.is_empty() {
                manifest.segments.retain(|s| s.visible_at_lsn <= cap_lsn);
            }
            removed
        };
        if !removed_seg_ids.is_empty() {
            // Also drop any already-loaded ColdSegment entries for the
            // removed IDs (rare at attach time but defensive — the
            // standalone-volume loader runs AFTER manifest load and
            // checks `manifest_has_segment` before reading from disk).
            let mut segments = self.segments.write();
            if removed_seg_ids.iter().any(|id| segments.contains_key(id)) {
                let mut new_map = (**segments).clone();
                for id in &removed_seg_ids {
                    new_map.remove(id);
                }
                *segments = Arc::new(new_map);
            }
        }

        // ---- Filter tombstones (V7) ----
        // Drop tombstones whose visibility frontier exceeds the cap.
        // V6 tombstones synthesized as `visible_at_lsn = 0` slip
        // through (V6 pre-dates capped attach and is treated as
        // always visible — that's the correct backward-compat
        // semantics: legacy data didn't have this discrimination).
        let removed_tomb_rids: Vec<i64> = {
            let mut manifest = self.manifest.write();
            let removed: Vec<i64> = manifest
                .tombstones
                .iter()
                .filter(|(_, _, vis)| *vis > cap_lsn)
                .map(|(rid, _, _)| *rid)
                .collect();
            if !removed.is_empty() {
                manifest.tombstones.retain(|(_, _, vis)| *vis <= cap_lsn);
            }
            removed
        };
        if !removed_tomb_rids.is_empty() {
            let mut ts_guard = self.tombstones.write();
            let ts = Arc::make_mut(&mut *ts_guard);
            for rid in &removed_tomb_rids {
                ts.remove(rid);
            }
        }

        if removed_seg_ids.is_empty() && removed_tomb_rids.is_empty() {
            return;
        }
        self.has_segments_flag.store(
            !self.manifest.read().segments.is_empty(),
            std::sync::atomic::Ordering::Relaxed,
        );
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get the volume directory path.
    pub fn volume_dir(&self) -> Option<&Path> {
        self.volume_dir.as_deref()
    }

    /// Recompute visibility bitmaps for all segments.
    /// Called after a batch of `load_volume_for_existing_segment` calls (recovery)
    /// so that the bitmaps reflect the final set of loaded volumes.
    pub fn recompute_visibility(&self) {
        let manifest = self.manifest.read();
        let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
        drop(manifest);
        let mut segments = self.segments.write();
        let mut new_map = (**segments).clone();
        compute_visibility_bitmaps(&seg_ids, &mut new_map, &mut self.visibility_seen.lock());
        *segments = Arc::new(new_map);
    }
}

impl std::fmt::Debug for SegmentManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let manifest = self.manifest.read();
        f.debug_struct("SegmentManager")
            .field("table", &self.table_name)
            .field("segments", &manifest.segments.len())
            .field("next_id", &manifest.next_segment_id)
            .field("tombstones", &self.tombstones.read().len())
            .finish()
    }
}

// Helper functions for binary deserialization — return errors on truncation
#[inline]
fn read_u32(data: &[u8], pos: &mut usize) -> std::io::Result<u32> {
    let end = *pos + 4;
    if end > data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated manifest: expected u32",
        ));
    }
    let v = u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
    *pos = end;
    Ok(v)
}

#[inline]
fn read_u64(data: &[u8], pos: &mut usize) -> std::io::Result<u64> {
    let end = *pos + 8;
    if end > data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated manifest: expected u64",
        ));
    }
    let v = u64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos = end;
    Ok(v)
}

#[inline]
fn read_i64(data: &[u8], pos: &mut usize) -> std::io::Result<i64> {
    let end = *pos + 8;
    if end > data.len() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "truncated manifest: expected i64",
        ));
    }
    let v = i64::from_le_bytes([
        data[*pos],
        data[*pos + 1],
        data[*pos + 2],
        data[*pos + 3],
        data[*pos + 4],
        data[*pos + 5],
        data[*pos + 6],
        data[*pos + 7],
    ]);
    *pos = end;
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manifest_new() {
        let m = TableManifest::new("test_table");
        assert_eq!(m.table_name.as_str(), "test_table");
        assert!(m.segments.is_empty());
        assert_eq!(m.next_segment_id, 1);
        assert_eq!(m.checkpoint_lsn, 0);
        assert!(m.tombstones.is_empty());
    }

    #[test]
    fn test_manifest_allocate_id() {
        let mut m = TableManifest::new("t");
        assert_eq!(m.allocate_segment_id(), 1);
        assert_eq!(m.allocate_segment_id(), 2);
        assert_eq!(m.allocate_segment_id(), 3);
        assert_eq!(m.next_segment_id, 4);
    }

    #[test]
    fn test_manifest_add_remove_segment() {
        let mut m = TableManifest::new("t");
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("vol_001.vol"),
            row_count: 1000,
            min_row_id: 1,
            max_row_id: 1000,
            creation_lsn: 100,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 0,
        });
        m.add_segment(SegmentMeta {
            segment_id: 2,
            file_path: PathBuf::from("vol_002.vol"),
            row_count: 500,
            min_row_id: 1001,
            max_row_id: 1500,
            creation_lsn: 200,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 0,
        });
        assert_eq!(m.segments.len(), 2);

        m.remove_segments(&[1]);
        assert_eq!(m.segments.len(), 1);
        assert_eq!(m.segments[0].segment_id, 2);
    }

    #[test]
    fn test_manifest_find_segment() {
        let mut m = TableManifest::new("t");
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("a.vol"),
            row_count: 100,
            min_row_id: 1,
            max_row_id: 100,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 0,
        });
        m.add_segment(SegmentMeta {
            segment_id: 2,
            file_path: PathBuf::from("b.vol"),
            row_count: 100,
            min_row_id: 101,
            max_row_id: 200,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 0,
        });

        assert_eq!(m.find_segment_for_row_id(50).unwrap().1.segment_id, 1);
        assert_eq!(m.find_segment_for_row_id(150).unwrap().1.segment_id, 2);
        assert!(m.find_segment_for_row_id(300).is_none());
    }

    #[test]
    fn segment_meta_is_visible_to_lsn_pin_zero_means_no_constraint() {
        // SWMR v2 Phase F: a writer-side scan or v1 reader (no LSN
        // pin) passes pinned_lsn = 0 and must see every segment.
        let s = SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("x"),
            row_count: 0,
            min_row_id: 0,
            max_row_id: 0,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 999_999,
        };
        assert!(
            s.is_visible_to_lsn(0),
            "pinned_lsn=0 (no pin) must see every segment"
        );
    }

    #[test]
    fn segment_meta_is_visible_to_lsn_filters_late_arrivals() {
        // SWMR v2 Phase F: a snapshot-pinned reader (pinned_lsn > 0)
        // sees segments with visible_at_lsn <= pinned_lsn but NOT
        // segments published after the pin was taken.
        let early = SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("e"),
            row_count: 0,
            min_row_id: 0,
            max_row_id: 0,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 100,
        };
        let late = SegmentMeta {
            visible_at_lsn: 200,
            ..early.clone()
        };
        assert!(early.is_visible_to_lsn(150), "early visible at pin=150");
        assert!(!late.is_visible_to_lsn(150), "late NOT visible at pin=150");
        assert!(late.is_visible_to_lsn(200), "late visible at its own LSN");
        assert!(late.is_visible_to_lsn(300), "late visible at pin > LSN");
    }

    #[test]
    fn manifest_v6_deserialize_synthesizes_visible_at_lsn_from_creation_lsn() {
        // SWMR v2 Phase F: a V6 manifest on disk must still load. The
        // writer upgrades to V7 on the next persist.
        //
        // Build a V6 manifest by hand to avoid needing an actual
        // pre-V7 binary.
        use std::io::Write as _;
        let mut buf: Vec<u8> = Vec::new();
        buf.write_all(&MANIFEST_MAGIC).unwrap();
        buf.write_all(&MANIFEST_VERSION_V6.to_le_bytes()).unwrap();
        buf.write_all(&5u64.to_le_bytes()).unwrap(); // next_segment_id
        buf.write_all(&42u64.to_le_bytes()).unwrap(); // checkpoint_lsn

        // Table name
        let name = b"v6_table";
        buf.write_all(&(name.len() as u32).to_le_bytes()).unwrap();
        buf.write_all(name).unwrap();

        // One segment in V6 layout (56 fixed bytes + variable path).
        buf.write_all(&1u32.to_le_bytes()).unwrap(); // seg count
        buf.write_all(&7u64.to_le_bytes()).unwrap(); // segment_id
        buf.write_all(&100u64.to_le_bytes()).unwrap(); // row_count
        buf.write_all(&1i64.to_le_bytes()).unwrap(); // min_row_id
        buf.write_all(&100i64.to_le_bytes()).unwrap(); // max_row_id
        buf.write_all(&555u64.to_le_bytes()).unwrap(); // creation_lsn
        buf.write_all(&0u64.to_le_bytes()).unwrap(); // seal_seq
        buf.write_all(&3u64.to_le_bytes()).unwrap(); // schema_version
                                                     // No visible_at_lsn in V6.
        let path = b"vol_007.vol";
        buf.write_all(&(path.len() as u32).to_le_bytes()).unwrap();
        buf.write_all(path).unwrap();

        // Empty tombstones / renames / dropped
        buf.write_all(&0u64.to_le_bytes()).unwrap(); // tombstone count
        buf.write_all(&0u32.to_le_bytes()).unwrap(); // rename count
        buf.write_all(&0u32.to_le_bytes()).unwrap(); // dropped count

        // Trailing CRC32 over the payload.
        let crc = crc32fast::hash(&buf);
        buf.write_all(&crc.to_le_bytes()).unwrap();

        let m = TableManifest::deserialize(&buf).expect("V6 manifest must parse");
        assert_eq!(m.table_name.as_str(), "v6_table");
        assert_eq!(m.segments.len(), 1);
        let s = &m.segments[0];
        assert_eq!(s.segment_id, 7);
        assert_eq!(s.creation_lsn, 555);
        assert_eq!(
            s.visible_at_lsn, s.creation_lsn,
            "V6 → V7 read must synthesize visible_at_lsn from creation_lsn"
        );

        // Re-serialize → V7 (current format). Verify it round-trips
        // and the version bytes flipped.
        let v7_bytes = m.serialize().unwrap();
        let version = u32::from_le_bytes(v7_bytes[4..8].try_into().unwrap());
        assert_eq!(version, MANIFEST_VERSION, "writer upgrades V6 → V7");
        let m2 = TableManifest::deserialize(&v7_bytes).expect("V7 round-trip");
        assert_eq!(m2.segments[0].visible_at_lsn, 555);
    }

    #[test]
    fn manifest_v7_round_trip_preserves_visible_at_lsn() {
        let mut m = TableManifest::new("v7_table");
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("a.vol"),
            row_count: 10,
            min_row_id: 1,
            max_row_id: 10,
            creation_lsn: 100,
            seal_seq: 0,
            schema_version: 1,
            visible_at_lsn: 200,
        });
        let bytes = m.serialize().unwrap();
        let m2 = TableManifest::deserialize(&bytes).unwrap();
        assert_eq!(m2.segments[0].visible_at_lsn, 200);
    }

    #[test]
    fn test_manifest_serialize_roundtrip() {
        let mut m = TableManifest::new("my_table");
        m.next_segment_id = 5;
        m.checkpoint_lsn = 42;
        m.tombstones = vec![(10, 0, 0), (20, 0, 0), (30, 0, 0)];
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("seg_0001.vol"),
            row_count: 10000,
            min_row_id: 1,
            max_row_id: 10000,
            creation_lsn: 10,
            seal_seq: 0,
            schema_version: 3,
            visible_at_lsn: 0,
        });
        m.add_segment(SegmentMeta {
            segment_id: 3,
            file_path: PathBuf::from("seg_0003.vol"),
            row_count: 5000,
            min_row_id: 10001,
            max_row_id: 15000,
            creation_lsn: 30,
            seal_seq: 0,
            schema_version: 5,
            visible_at_lsn: 0,
        });

        let data = m.serialize().unwrap();
        let loaded = TableManifest::deserialize(&data).unwrap();

        assert_eq!(loaded.table_name.as_str(), "my_table");
        assert_eq!(loaded.next_segment_id, 5);
        assert_eq!(loaded.checkpoint_lsn, 42);
        assert_eq!(loaded.segments.len(), 2);
        assert_eq!(loaded.segments[0].segment_id, 1);
        assert_eq!(loaded.segments[0].row_count, 10000);
        assert_eq!(loaded.segments[0].min_row_id, 1);
        assert_eq!(loaded.segments[0].max_row_id, 10000);
        assert_eq!(loaded.segments[0].file_path, PathBuf::from("seg_0001.vol"));
        assert_eq!(loaded.segments[1].segment_id, 3);
        assert_eq!(loaded.segments[1].creation_lsn, 30);
        assert_eq!(loaded.segments[0].schema_version, 3);
        assert_eq!(loaded.segments[1].schema_version, 5);
        assert_eq!(loaded.tombstones, vec![(10, 0, 0), (20, 0, 0), (30, 0, 0)]);
    }

    #[test]
    fn test_manifest_disk_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.bin");

        let mut m = TableManifest::new("disk_test");
        m.tombstones = vec![(5, 0, 0), (10, 0, 0)];
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("vol.vol"),
            row_count: 100,
            min_row_id: 1,
            max_row_id: 100,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 0,
        });

        m.write_to_disk(&path).unwrap();
        let loaded = TableManifest::read_from_disk(&path).unwrap();

        assert_eq!(loaded.table_name.as_str(), "disk_test");
        assert_eq!(loaded.segments.len(), 1);
        assert_eq!(loaded.tombstones, vec![(5, 0, 0), (10, 0, 0)]);
    }

    #[test]
    fn test_segment_manager_register_and_query() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        let mut builder = super::super::writer::VolumeBuilder::new(&schema);
        for i in 1..=10i64 {
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
        }
        let volume = Arc::new(builder.finish());

        let mgr = SegmentManager::new("test", None);
        let meta = SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("test.vol"),
            row_count: 10,
            min_row_id: 1,
            max_row_id: 10,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 0,
        };
        mgr.register_segment(1, volume, meta, None);

        assert_eq!(mgr.segment_count(), 1);
        assert_eq!(mgr.total_row_count(), 10);
        assert!(mgr.row_exists(5));
        assert!(!mgr.row_exists(11));
    }

    #[test]
    fn test_segment_manager_tombstones() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        let mut builder = super::super::writer::VolumeBuilder::new(&schema);
        for i in 1..=10i64 {
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
        }
        let volume = Arc::new(builder.finish());

        let mgr = SegmentManager::new("test", None);
        mgr.register_segment(
            1,
            volume,
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("test.vol"),
                row_count: 10,
                min_row_id: 1,
                max_row_id: 10,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
                visible_at_lsn: 0,
            },
            None,
        );

        // Tombstone row_id=5 (commit_seq=1)
        mgr.add_tombstones(&[5], 1, 0);
        assert!(!mgr.row_exists(5));
        assert!(mgr.row_exists(4));
        assert!(mgr.row_exists(6));
        // total_row_count is the upper-bound row hint and does NOT
        // subtract tombstones (they may be orphans from previously-
        // compacted segments). The exact count comes from
        // deduped_row_count.
        assert_eq!(mgr.total_row_count(), 10);
        assert_eq!(mgr.deduped_row_count(), 9);
        assert!(mgr.is_tombstoned(5));
        assert!(!mgr.is_tombstoned(4));

        // Add an ORPHAN tombstone (row_id 999 isn't in any segment).
        // This is the regression case: total_row_count must NOT
        // shrink to 8; it must stay at 10. Pre-fix, total_row_count
        // blindly subtracted tombstones.len() and would return 8,
        // which fooled planner / cache hints into treating large
        // tables as small after compaction left orphans behind.
        mgr.add_tombstones(&[999], 2, 0);
        assert_eq!(mgr.total_row_count(), 10);
        assert_eq!(mgr.deduped_row_count(), 9);

        // Clear tombstones
        mgr.clear_tombstones();
        assert!(mgr.row_exists(5));
        assert_eq!(mgr.total_row_count(), 10);
        assert_eq!(mgr.deduped_row_count(), 10);
    }

    #[test]
    fn test_segment_manager_volumes_newest_first() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("test")
            .column("id", DataType::Integer, false, true)
            .build();

        let mgr = SegmentManager::new("test", None);

        for seg_id in [1u64, 3, 2] {
            let mut builder = super::super::writer::VolumeBuilder::new(&schema);
            builder.add_row(
                seg_id as i64,
                &Row::from_values(vec![Value::Integer(seg_id as i64)]),
            );
            let vol = Arc::new(builder.finish());
            mgr.register_segment(
                seg_id,
                vol,
                SegmentMeta {
                    segment_id: seg_id,
                    file_path: PathBuf::from(format!("vol_{}.vol", seg_id)),
                    row_count: 1,
                    min_row_id: seg_id as i64,
                    max_row_id: seg_id as i64,
                    creation_lsn: 0,
                    seal_seq: 0,
                    schema_version: 0,
                    visible_at_lsn: 0,
                },
                None,
            );
        }

        let newest_first = mgr.get_volumes_newest_first();
        assert_eq!(newest_first.len(), 3);
        // get_volumes_newest_first reverses manifest insertion order.
        // Segments were registered as [1, 3, 2], so reversed = [2, 3, 1].
        assert_eq!(newest_first[0].0, 2);
        assert_eq!(newest_first[1].0, 3);
        assert_eq!(newest_first[2].0, 1);
    }

    #[test]
    fn test_segment_manager_clear() {
        let mgr = SegmentManager::new("test", None);
        mgr.manifest.write().add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("x.vol"),
            row_count: 10,
            min_row_id: 1,
            max_row_id: 10,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
            visible_at_lsn: 0,
        });
        mgr.add_tombstones(&[5], 1, 0);

        assert_eq!(mgr.segment_count(), 1);
        mgr.clear();
        assert_eq!(mgr.segment_count(), 0);
        assert!(mgr.tombstone_set_arc().is_empty());
    }

    #[test]
    fn test_eviction_lifecycle() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("evict_test")
            .column("id", DataType::Integer, false, true)
            .column("name", DataType::Text, false, false)
            .build();

        let mgr = SegmentManager::new("evict_test", None);

        // Register a volume (simulates seal). VolumeBuilder produces hot (eager) volumes.
        let mut builder = super::super::writer::VolumeBuilder::new(&schema);
        for i in 1..=100i64 {
            builder.add_row(
                i,
                &Row::from_values(vec![Value::Integer(i), Value::from(format!("name_{}", i))]),
            );
        }
        let mut volume = builder.finish();
        // Attach compressed store so hot→warm transition works.
        let (_, store) = crate::storage::volume::io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        let volume = Arc::new(volume);

        mgr.register_segment(
            1,
            volume,
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("test.vol"),
                row_count: 100,
                min_row_id: 1,
                max_row_id: 100,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
                visible_at_lsn: 0,
            },
            None,
        );

        // Verify: volume starts hot (eager + compressed store).
        {
            let segs = mgr.segments_raw();
            let cs = segs.get(&1).unwrap();
            assert!(cs.volume.columns.is_eager(), "should start hot");
            assert!(
                cs.volume.columns.has_compressed_store(),
                "should have compressed store"
            );
            assert!(!cs.volume.is_warm(), "hot is not warm");
            assert!(!cs.volume.is_cold(), "hot is not cold");
        }

        // Data is correct while hot.
        {
            let vols = mgr.get_volumes_newest_first();
            let (_, cs) = &vols[0];
            let row = cs.volume.get_row(0);
            assert_eq!(row[0], Value::Integer(1));
        }

        // Helper: mirror engine's evict_idle_volumes behavior.
        // Engine does: fetch_add → GLOBAL.fetch_max → mgr.evict_idle_volumes.
        let run_eviction = |mgr: &SegmentManager, epoch: u64| {
            super::super::writer::GLOBAL_EVICTION_EPOCH
                .fetch_max(epoch, std::sync::atomic::Ordering::Relaxed);
            mgr.evict_idle_volumes(epoch);
        };

        // ── Eviction cycle 0..2: not enough idle cycles, no eviction ──
        for epoch in 0..3u64 {
            run_eviction(&mgr, epoch);
            let segs = mgr.segments_raw();
            let cs = segs.get(&1).unwrap();
            assert!(
                cs.volume.columns.is_eager(),
                "epoch {}: should still be hot (< MIN_IDLE_CYCLES)",
                epoch
            );
        }

        // ── Eviction cycle 3: idle for 3 cycles → hot → warm ──
        run_eviction(&mgr, 3);
        {
            let segs = mgr.segments_raw();
            let cs = segs.get(&1).unwrap();
            assert!(
                cs.volume.is_warm(),
                "epoch 3: should be warm after eviction"
            );
            assert!(
                !cs.volume.columns.is_eager(),
                "epoch 3: warm means not eager"
            );
            assert!(
                cs.volume.columns.has_compressed_store(),
                "epoch 3: warm still has compressed store"
            );
        }

        // Data is correct while warm (decompresses from compressed store).
        {
            let vols = mgr.get_volumes_newest_first();
            let (_, cs) = &vols[0];
            let row = cs.volume.get_row(49);
            assert_eq!(row[0], Value::Integer(50));
        }

        // ── Mark accessed (simulates a query) → should NOT be evicted ──
        // get_volumes_newest_first marks all volumes.
        let _ = mgr.get_volumes_newest_first();

        // Eviction cycles 4..6: volume was accessed, should stay warm/hot.
        for epoch in 4..7u64 {
            run_eviction(&mgr, epoch);
            let segs = mgr.segments_raw();
            let cs = segs.get(&1).unwrap();
            assert!(
                !cs.volume.is_cold(),
                "epoch {}: actively queried volume should NOT go cold",
                epoch
            );
        }

        // ── Stop querying. After idle cycles: should eventually go cold ──
        // Epoch 7: the volume was marked u64::MAX by get_volumes_newest_first.
        // Eviction at epoch 4 reset it to 4. At epoch 7: delta = 3 → evict
        // hot→warm (OnceLock filled by get_row, so is_eager=true). This is the
        // first demotion. The new warm volume starts at GLOBAL (7).
        run_eviction(&mgr, 7);
        // Epochs 8, 9: delta grows from warm volume's start epoch (7)
        run_eviction(&mgr, 8);
        run_eviction(&mgr, 9);
        {
            let segs = mgr.segments_raw();
            let cs = segs.get(&1).unwrap();
            assert!(!cs.volume.is_cold(), "epoch 9: delta=2, still warm");
        }
        // Epoch 10: delta=3 → warm → cold
        run_eviction(&mgr, 10);
        {
            let segs = mgr.segments_raw();
            let cs = segs.get(&1).unwrap();
            assert!(
                cs.volume.is_cold(),
                "epoch 10: should be cold after 3 idle cycles"
            );
            assert!(
                !cs.volume.columns.has_compressed_store(),
                "cold has no compressed store"
            );
        }

        // Metadata still accessible on cold volumes.
        assert!(mgr.row_exists(50), "metadata (row_ids) should work on cold");
        assert_eq!(mgr.total_row_count(), 100);

        // Volume is still in the map (not removed).
        assert_eq!(mgr.segment_count(), 1);
    }
}
