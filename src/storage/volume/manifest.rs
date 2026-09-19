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

use crate::storage::mvcc::get_fast_timestamp;

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
    /// The handle of this volume's file, taken when the segment is
    /// registered. A reader holds it with the segment, so a compaction that
    /// retires the file cannot take it from the reader, and reading goes
    /// through the handle rather than a path resolved per statement. The
    /// registry moves the path inside it on a rename. None without a
    /// volume directory, where there are no files.
    pub file: Option<Arc<super::writer::VolumeFile>>,
}

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

/// A reader's cold view. It holds what `ColdSnapshot` holds and, taken under
/// the same locks, the handle of every volume's file: a compaction that
/// retires one of those files cannot take it from this reader, and the read
/// that follows goes through the handle rather than the path.
pub struct PinnedColdView {
    pub seg_ids: smallvec::SmallVec<[u64; 4]>,
    pub segs: Arc<FxHashMap<u64, ColdSegment>>,
    pub ts: Arc<FxHashMap<i64, u64>>,
}

impl PinnedColdView {
    /// The handle of the file the volume `seg_id` reads from. The captured
    /// segment holds it, so it lives as long as this view does.
    pub fn file(&self, seg_id: u64) -> Option<Arc<super::writer::VolumeFile>> {
        self.segs.get(&seg_id)?.file.clone()
    }

    /// The volumes, newest first, as the reads below want them.
    pub fn volumes(&self) -> Vec<(u64, ColdSegment)> {
        self.seg_ids
            .iter()
            .filter_map(|&id| self.segs.get(&id).map(|cs| (id, cs.clone())))
            .collect()
    }
}

// Manifest file magic: "STMF" (SToolap ManiFest)
const MANIFEST_MAGIC: [u8; 4] = *b"STMF";
const MANIFEST_VERSION: u32 = 6;

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
    /// Tombstone entries: (row_id, commit_seq) pairs for cold rows that have
    /// been deleted or superseded. The commit_seq records when the tombstone
    /// was created, enabling snapshot isolation: a snapshot transaction at
    /// begin_seq=N only sees tombstones with commit_seq <= N.
    /// Cleared after compaction processes them.
    pub tombstones: Vec<(i64, u64)>,
    /// Column rename history: (old_name, new_name) pairs.
    /// Applied as aliases to cold volumes on load so pre-rename data
    /// is visible through the new schema column name.
    pub column_renames: Vec<(SmartString, SmartString)>,
    /// Columns that have been dropped (and possibly re-added with same name).
    /// Each entry is (column_name, schema_version_at_drop). Old volumes sealed
    /// before the drop (schema_version <= drop_version) have stale data masked.
    /// Cleared during compaction (new volumes don't have stale data).
    pub dropped_columns: Vec<(SmartString, u64)>,
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

        // Table name
        let name_bytes = self.table_name.as_bytes();
        buf.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        buf.write_all(name_bytes)?;

        // Segments
        buf.write_all(&(self.segments.len() as u32).to_le_bytes())?;
        for seg in &self.segments {
            buf.write_all(&seg.segment_id.to_le_bytes())?;
            buf.write_all(&(seg.row_count as u64).to_le_bytes())?;
            buf.write_all(&seg.min_row_id.to_le_bytes())?;
            buf.write_all(&seg.max_row_id.to_le_bytes())?;
            buf.write_all(&seg.creation_lsn.to_le_bytes())?;
            buf.write_all(&seg.seal_seq.to_le_bytes())?;
            buf.write_all(&seg.schema_version.to_le_bytes())?;

            // File path as UTF-8 string
            let path_str = seg.file_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            buf.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
            buf.write_all(path_bytes)?;
        }

        // Tombstones: (row_id, commit_seq) pairs
        buf.write_all(&(self.tombstones.len() as u64).to_le_bytes())?;
        for &(row_id, commit_seq) in &self.tombstones {
            buf.write_all(&row_id.to_le_bytes())?;
            buf.write_all(&commit_seq.to_le_bytes())?;
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

    /// Deserialize a manifest from bytes. Only V6 format is supported.
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
        if version != MANIFEST_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unsupported manifest version {} (expected {})",
                    version, MANIFEST_VERSION
                ),
            ));
        }

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

        // Segments: V6 fixed fields are 56 bytes per segment (before variable-length path)
        let seg_count = read_u32(data, &mut pos)? as usize;
        let mut segments = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            if pos + 56 > data.len() {
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
            });
        }

        // Last 4 bytes are CRC, so stop before them.
        let data_end = data.len() - 4;

        // Tombstones: (row_id: i64, commit_seq: u64) pairs
        let mut tombstones = Vec::new();
        if pos + 8 <= data_end {
            let tombstone_count = read_u64(data, &mut pos)? as usize;
            tombstones.reserve(tombstone_count);
            for _ in 0..tombstone_count {
                if pos + 16 > data_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "manifest truncated at tombstone entry",
                    ));
                }
                let row_id = read_i64(data, &mut pos)?;
                let commit_seq = read_u64(data, &mut pos)?;
                tombstones.push((row_id, commit_seq));
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
            super::io::sync_ordered(&f).map_err(|e| {
                crate::core::Error::internal(format!("failed to fsync manifest: {}", e))
            })?;
        }
        std::fs::rename(&tmp_path, path).map_err(|e| {
            crate::core::Error::internal(format!("failed to rename manifest: {}", e))
        })?;
        // The full sync of the directory makes the rename durable, and
        // with it everything the publication ordered before this point.
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

/// A publication built outside the locks: the segments it was built on,
/// their manifest order then, and the map to swap in
struct PreparedPublication {
    snapshot: Arc<FxHashMap<u64, ColdSegment>>,
    order: Vec<u64>,
    map: FxHashMap<u64, ColdSegment>,
}

/// The manifest order after `old` leave and `new` take the first one's place
fn published_order(
    order: &[u64],
    new_volumes: &[(u64, Arc<FrozenVolume>, SegmentMeta)],
    old: &[u64],
) -> Vec<u64> {
    let insert_pos = order
        .iter()
        .position(|id| old.contains(id))
        .unwrap_or(order.len());
    let mut kept: Vec<u64> = order
        .iter()
        .copied()
        .filter(|id| !old.contains(id))
        .collect();
    let insert_pos = insert_pos.min(kept.len());
    for (i, (id, _, _)) in new_volumes.iter().enumerate() {
        kept.insert(insert_pos + i, *id);
    }
    kept
}

/// Per-table segment manager.
///
/// The mapping a segment is published with: computed against `schema`
/// from the manifest's history and the segment's own version, or the
/// identity when the caller vouches the volume matches the schema. Taken
/// under the manifest lock the publication holds, so the mapping and the
/// segment appear together
fn mapping_for(
    manifest: &TableManifest,
    volume: &FrozenVolume,
    schema_version: u64,
    schema: Option<&crate::core::Schema>,
) -> super::writer::ColumnMapping {
    match schema {
        Some(schema) => super::writer::compute_column_mapping_with_drops(
            schema,
            volume,
            &manifest.dropped_columns,
            schema_version,
            &manifest.column_renames,
        ),
        None => super::writer::ColumnMapping {
            sources: (0..volume.columns.len())
                .map(super::writer::ColSource::Volume)
                .collect(),
            names: Vec::new(),
            is_identity: true,
        },
    }
}

/// Owns the manifest, loaded segments, and tombstone set for one table.
/// Tombstones track cold row_ids that have been deleted or superseded
/// by hot buffer versions. They are persisted in the manifest and used
/// during scans to skip stale cold rows.
///
/// Thread safety: the manager uses interior mutability via RwLock for
/// concurrent read access (queries) and exclusive write access (seal, compaction).
/// Statement-scoped consistent view of cold state (see
/// `SegmentManager::statement_snapshot`).
pub struct StatementSnapshot {
    /// Manifest segment ids, newest first, captured with `segs`.
    pub seg_ids_newest_first: Vec<u64>,
    pub segs: Arc<FxHashMap<u64, ColdSegment>>,
    /// Committed tombstones as of capture.
    pub tombstones: Arc<FxHashMap<i64, u64>>,
}

impl StatementSnapshot {
    /// Newest-first (id, segment) list from the snapshot alone.
    pub fn volumes_newest_first(&self) -> Vec<(u64, ColdSegment)> {
        self.seg_ids_newest_first
            .iter()
            .filter_map(|&id| self.segs.get(&id).map(|cs| (id, cs.clone())))
            .collect()
    }
}

/// Keeps a destructive publication marked for as long as it runs, including
/// on the error path.
pub struct Destruction<'a> {
    mgr: &'a SegmentManager,
}

impl Drop for Destruction<'_> {
    fn drop(&mut self) {
        self.mgr
            .destruction_count
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    }
}

pub struct SegmentManager {
    /// Table name; a rename changes it under every open handle
    table_name: RwLock<SmartString>,
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
    /// Recluster requests, one at construction and one per key change, and
    /// the last request a full check of the volumes closed
    recluster_requested: std::sync::atomic::AtomicU64,
    recluster_done: std::sync::atomic::AtomicU64,
    /// Per segment, the volume columns checked for order and the verdict;
    /// volumes are immutable, so a verdict stands until the segment goes
    key_order_verdicts: parking_lot::Mutex<FxHashMap<u64, (Vec<usize>, bool)>>,
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
    /// Each row_id carries the timestamp it was tombstoned at, so a savepoint
    /// rollback can discard the tombstones made after the savepoint.
    pending_txn_tombstones: RwLock<FxHashMap<i64, FxHashMap<i64, i64>>>,
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
    /// Destructive publications in flight. Unlike the overlap count, this
    /// says nothing about how many rows moved: it marks the interval in which
    /// rows are leaving, so a reader that captured inside it is not answered.
    destruction_count: std::sync::atomic::AtomicUsize,
}

impl SegmentManager {
    /// Create a new segment manager for a table.
    pub fn new(table_name: &str, volume_dir: Option<PathBuf>) -> Self {
        Self {
            table_name: RwLock::new(SmartString::from(table_name)),
            manifest: RwLock::new(TableManifest::new(table_name)),
            segments: RwLock::new(Arc::new(FxHashMap::default())),
            volume_dir,
            has_segments_flag: std::sync::atomic::AtomicBool::new(false),
            has_cold: std::sync::atomic::AtomicBool::new(false),
            recluster_requested: std::sync::atomic::AtomicU64::new(1),
            recluster_done: std::sync::atomic::AtomicU64::new(0),
            key_order_verdicts: parking_lot::Mutex::new(FxHashMap::default()),
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
            destruction_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Create from an existing manifest loaded from disk.
    pub fn from_manifest(manifest: TableManifest, volume_dir: Option<PathBuf>) -> Self {
        let table_name = manifest.table_name.clone();
        let tombstone_map: FxHashMap<i64, u64> = manifest.tombstones.iter().copied().collect();
        Self {
            table_name: RwLock::new(table_name),
            manifest: RwLock::new(manifest),
            segments: RwLock::new(Arc::new(FxHashMap::default())),
            volume_dir,
            has_segments_flag: std::sync::atomic::AtomicBool::new(false),
            has_cold: std::sync::atomic::AtomicBool::new(false),
            recluster_requested: std::sync::atomic::AtomicU64::new(1),
            recluster_done: std::sync::atomic::AtomicU64::new(0),
            key_order_verdicts: parking_lot::Mutex::new(FxHashMap::default()),
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
            destruction_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Get the table name.
    pub fn table_name(&self) -> SmartString {
        self.table_name.read().clone()
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
    pub fn get_volumes_newest_first(&self) -> crate::core::Result<Arc<Vec<(u64, ColdSegment)>>> {
        self.ensure_columns();
        let mut result = self.build_volumes_newest_first();
        // Race check: eviction may have created cold volumes between
        // ensure_columns (fast-path has_cold=false) and segments.read().
        // Retry once — eviction runs once per ~30s checkpoint cycle.
        if result.iter().any(|(_, cs)| cs.volume.is_cold()) {
            self.ensure_columns();
            result = self.build_volumes_newest_first();
            // Still cold after retry = persistent reload failure. Serving
            // the surviving subset silently returns short scan results and
            // lets DML skip rows while reporting success; fail closed.
            let cold: Vec<(u64, usize)> = result
                .iter()
                .filter(|(_, cs)| cs.volume.is_cold())
                .map(|(id, cs)| (*id, cs.volume.meta.row_count))
                .collect();
            if !cold.is_empty() {
                return Err(crate::core::Error::Internal {
                    message: format!(
                        "table '{}': cold volume reload failed for segment(s) {:?}; \
                         refusing to serve partial data",
                        self.table_name.read(),
                        cold.iter().map(|(id, _)| *id).collect::<Vec<_>>()
                    ),
                });
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
        Ok(Arc::new(result))
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

    /// Capture a consistent, statement-scoped view of the cold state:
    /// newest-first segment ids, the segment map, and the committed
    /// tombstone set, all under ONE manifest lock hold, with every volume
    /// verified warm (fail-closed like `segments_snapshot`). DML uses this
    /// for the whole statement and never re-reads live manager state after
    /// mutating the hot buffer: eviction and compaction copy-on-write new
    /// maps/Arcs, so nothing in the snapshot can go cold or disappear.
    /// Statement-snapshot variant of `find_row_id_by_values`: runs
    /// entirely against the given captured parts, so it can neither
    /// observe concurrent compaction nor need a live cold reload.
    pub fn find_row_id_by_values_in(
        &self,
        seg_ids: &[u64],
        segs: &FxHashMap<u64, ColdSegment>,
        ts: &FxHashMap<i64, u64>,
        col_indices: &[usize],
        values: &[&Value],
        column_defaults: &[Value],
    ) -> crate::core::Result<Option<i64>> {
        self.find_row_id_by_values_impl(seg_ids, segs, ts, col_indices, values, column_defaults)
    }

    pub fn statement_snapshot(&self) -> crate::core::Result<StatementSnapshot> {
        let capture = || {
            let manifest = self.manifest.read();
            let mut seg_ids_newest_first: Vec<u64> =
                manifest.segments.iter().map(|m| m.segment_id).collect();
            seg_ids_newest_first.reverse();
            let segs = Arc::clone(&*self.segments.read());
            let tombstones = Arc::clone(&*self.tombstones.read());
            StatementSnapshot {
                seg_ids_newest_first,
                segs,
                tombstones,
            }
        };
        self.ensure_columns();
        let mut snap = capture();
        if snap.segs.values().any(|cs| cs.volume.is_cold()) {
            // Race with eviction: retry once, then fail closed.
            self.ensure_columns();
            snap = capture();
            let cold: Vec<u64> = snap
                .segs
                .iter()
                .filter(|(_, cs)| cs.volume.is_cold())
                .map(|(&id, _)| id)
                .collect();
            if !cold.is_empty() {
                return Err(crate::core::Error::Internal {
                    message: format!(
                        "table '{}': cold volume reload failed for segment(s) {:?}; \
                         refusing to serve partial data",
                        self.table_name.read(),
                        cold
                    ),
                });
            }
        }
        Ok(snap)
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

    /// A reader's cold view: the one-hold snapshot, whose segments carry the
    /// handles of their files. A read resolves nothing here, so it costs no
    /// path building, no registry lookup and no reload guard.
    pub fn pinned_cold_view(&self) -> PinnedColdView {
        let snapshot = self.cold_snapshot();
        PinnedColdView {
            seg_ids: snapshot.seg_ids,
            segs: snapshot.segs,
            ts: snapshot.ts,
        }
    }

    /// The handle of a segment's file, resolved the way `ensure_volume`
    /// resolves it: the volume directory, the table's current name and the
    /// segment id. Only a memory-backed volume needs this; a file-backed one
    /// is registered with the handle it was opened with.
    pub(crate) fn file_of(&self, seg_id: u64) -> Option<Arc<super::writer::VolumeFile>> {
        let dir = self.volume_dir.as_ref()?;
        let path = dir
            .join(self.table_name.read().as_str())
            .join(format!("vol_{:016x}.vol", seg_id));
        Some(super::writer::VolumeFile::shared(&path))
    }

    /// Check if a value exists using a pre-captured snapshot (no lock acquisition).
    pub fn check_value_exists_with_snapshot(
        &self,
        snapshot: &ColdSnapshot,
        col_idx: usize,
        value: &crate::core::Value,
    ) -> crate::core::Result<Option<i64>> {
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
    ) -> crate::core::Result<Option<i64>> {
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
    /// Resolve against the CALLER'S segment view (newest-first ids + map)
    /// so statement-snapshot flows never re-read live state here.
    fn get_authoritative_value(
        &self,
        seg_ids: &[u64],
        segments: &FxHashMap<u64, ColdSegment>,
        row_id: i64,
        col_idx: usize,
    ) -> crate::core::Result<Option<crate::core::Value>> {
        for seg_id in seg_ids {
            if let Some(cold) = segments.get(seg_id) {
                if let Some(idx) = cold.volume.locate(row_id) {
                    let pi = if cold.mapping.is_identity {
                        col_idx
                    } else if col_idx < cold.mapping.sources.len() {
                        match &cold.mapping.sources[col_idx] {
                            super::writer::ColSource::Volume(vi) => *vi,
                            super::writer::ColSource::Default(val) => return Ok(Some(val.clone())),
                        }
                    } else {
                        return Ok(None);
                    };
                    if cold.volume.is_cold() {
                        if let Some(vol) = self.ensure_volume(*seg_id)? {
                            return Ok(Some(vol.cell(pi, idx)?));
                        }
                        return Ok(None);
                    }
                    cold.volume.mark_accessed();
                    return Ok(Some(cold.volume.cell(pi, idx)?));
                }
            }
        }
        Ok(None)
    }

    fn check_value_exists_impl(
        &self,
        seg_ids: &[u64],
        segs: &FxHashMap<u64, ColdSegment>,
        ts: &FxHashMap<i64, u64>,
        col_idx: usize,
        value: &crate::core::Value,
    ) -> crate::core::Result<Option<i64>> {
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
                loaded = match self.ensure_volume(seg_id)? {
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
                let row_ids = vol.row_ids()?;
                // The column is read a group at a time; a sorted column
                // from the group its first candidate is in
                let first = if vol.is_sorted(pi) {
                    match vol.first_index_ge(pi, target)? {
                        Some(first) => first,
                        None => continue,
                    }
                } else {
                    0
                };
                let mut found: Option<i64> = None;
                vol.for_each_group_from::<crate::core::Error>(pi, first, |start, col| {
                    let local_first = first.saturating_sub(start);
                    for i in local_first..col.len() {
                        let global = start + i;
                        if vol.is_sorted(pi) && (col.is_null(i) || col.get_i64(i) != target) {
                            return Ok(false);
                        }
                        let rid = row_ids[global];
                        if !seen.insert(rid) {
                            continue;
                        }
                        if !col.is_null(i) && col.get_i64(i) == target && !ts.contains_key(&rid) {
                            if seg_ids.len() > 1 {
                                if let Some(current_val) =
                                    self.get_authoritative_value(seg_ids, segs, rid, col_idx)?
                                {
                                    if &current_val != value {
                                        continue;
                                    }
                                }
                            }
                            found = Some(rid);
                            return Ok(false);
                        }
                    }
                    Ok(true)
                })?;
                if found.is_some() {
                    return Ok(found);
                }
            }
        }
        Ok(None)
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
    ) -> crate::core::Result<Option<i64>> {
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
    ) -> crate::core::Result<Option<i64>> {
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
    ) -> crate::core::Result<Option<i64>> {
        if col_indices.is_empty() || col_indices.len() != values.len() {
            return Ok(None);
        }

        if seg_ids.is_empty() {
            return Ok(None);
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
                loaded = match self.ensure_volume(seg_id)? {
                    Some(v) => v,
                    None => continue,
                };
                &*loaded
            } else {
                vol.mark_accessed();
                vol
            };
            // Tier 3: Per-volume hash index
            let row_ids = vol.row_ids()?;
            let mut vol_result: Option<i64> = None;
            if !has_missing {
                // Common path: no schema evolution, pass values directly (zero alloc)
                vol.unique_lookup_all(&vol_col_indices, values, |row_idx| {
                    let rid = row_ids[row_idx as usize];
                    if ts.contains_key(&rid) {
                        false
                    } else if seen.insert(rid) {
                        vol_result = Some(rid);
                        true
                    } else {
                        false
                    }
                })?;
            } else {
                // Schema-evolved volume: some columns missing (default matches).
                // Check only the columns that exist in the volume, cell by
                // cell through the row's group
                let present: Vec<(usize, usize)> = vol_col_indices
                    .iter()
                    .enumerate()
                    .filter(|(_, &vi)| vi != usize::MAX)
                    .map(|(i, &vi)| (i, vi))
                    .collect();
                for (i, &rid) in row_ids.iter().enumerate() {
                    if ts.contains_key(&rid) || !seen.insert(rid) {
                        continue;
                    }
                    let mut matches = true;
                    for &(val_idx, vi) in &present {
                        let v = vol.cell(vi, i)?;
                        if v.is_null() || v != *values[val_idx] {
                            matches = false;
                            break;
                        }
                    }
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
                    let mut still_matches = true;
                    for (i, &ci) in col_indices.iter().enumerate() {
                        let ok = match self.get_authoritative_value(seg_ids, segs, rid, ci)? {
                            Some(v) => !v.is_null() && v == *values[i],
                            None => column_defaults[i] == *values[i],
                        };
                        if !ok {
                            still_matches = false;
                            break;
                        }
                    }
                    if !still_matches {
                        continue; // stale value in older volume, skip
                    }
                }
                return Ok(Some(rid));
            }
        }
        Ok(None)
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
                } else if let Some(store) = vol.columns.compressed_store() {
                    if store.is_file_backed() {
                        "file"
                    } else {
                        "warm"
                    }
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
            let full_path = vol_dir.join(self.table_name.read().as_str()).join(filename);
            match crate::storage::volume::io::read_volume_from_disk(&full_path) {
                Ok(volume) => {
                    reloaded.push((id, Arc::new(volume)));
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to reload cold volume {} seg={}: {}",
                        self.table_name.read(),
                        id,
                        e
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
                volume.inherit_row_order(&cs.volume);
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

    /// The recluster request still open, if any: not every volume has been
    /// checked against the table's key since the request was opened
    pub fn recluster_request(&self) -> Option<u64> {
        let requested = self
            .recluster_requested
            .load(std::sync::atomic::Ordering::Acquire);
        let done = self
            .recluster_done
            .load(std::sync::atomic::Ordering::Acquire);
        (requested != done).then_some(requested)
    }

    /// Whether a recluster request is open
    pub fn recluster_pending(&self) -> bool {
        self.recluster_request().is_some()
    }

    /// Open a request: the key changed
    pub fn request_recluster(&self) {
        self.recluster_requested
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    }

    /// Close `request`: every volume was checked against the key it was
    /// opened for. A request opened since stays open
    pub fn recluster_done(&self, request: u64) {
        self.recluster_done
            .fetch_max(request, std::sync::atomic::Ordering::AcqRel);
    }

    /// The key as the volume of `seg_id` holds it: each schema column
    /// through the volume's column mapping. A column the volume predates
    /// holds one value for every row and orders nothing, so it is left
    /// out. None when the segment is not registered
    fn volume_key(&self, seg_id: u64, key: &[usize]) -> Option<Vec<usize>> {
        use super::writer::ColSource;
        let segs = self.segments.read();
        let mapping = &segs.get(&seg_id)?.mapping;
        Some(
            key.iter()
                .filter_map(|&column| {
                    if mapping.is_identity {
                        return Some(column);
                    }
                    match mapping.sources.get(column) {
                        Some(ColSource::Volume(v)) => Some(*v),
                        _ => None,
                    }
                })
                .collect(),
        )
    }

    /// The verdict decided earlier for the volume of `seg_id` under `key`
    pub fn known_key_order(&self, seg_id: u64, key: &[usize]) -> Option<bool> {
        let columns = self.volume_key(seg_id, key)?;
        self.key_order_verdicts
            .lock()
            .get(&seg_id)
            .and_then(|(checked, verdict)| (*checked == columns).then_some(*verdict))
    }

    /// Decide whether the volume of `seg_id` holds its rows in `key` order,
    /// loading a cold volume to do it, and keep the verdict; a segment no
    /// longer registered is in order
    pub fn decide_key_order(&self, seg_id: u64, key: &[usize]) -> crate::core::Result<bool> {
        let Some(columns) = self.volume_key(seg_id, key) else {
            return Ok(true);
        };
        let volume = {
            let segs = self.segments.read();
            segs.get(&seg_id).map(|cs| Arc::clone(&cs.volume))
        };
        let volume = match volume {
            Some(v) if !v.is_cold() => Some(v),
            Some(_) => self.ensure_volume(seg_id)?,
            None => None,
        };
        let Some(volume) = volume else {
            return Ok(true);
        };
        let verdict = volume.in_key_order(&columns)?;
        self.key_order_verdicts
            .lock()
            .insert(seg_id, (columns, verdict));
        Ok(verdict)
    }

    /// Keep that the volume of `seg_id` was written in the order of its own
    /// `columns`: the physical positions, which the writer knows, not a
    /// schema key, which the mapping current at the time would resolve
    /// against a schema the volume may no longer match
    pub fn record_key_order(&self, seg_id: u64, columns: &[usize]) {
        self.key_order_verdicts
            .lock()
            .insert(seg_id, (columns.to_vec(), true));
    }

    /// Drop the verdicts of segments that are gone
    fn forget_key_order(&self, segment_ids: &[u64]) {
        let mut verdicts = self.key_order_verdicts.lock();
        for id in segment_ids {
            verdicts.remove(id);
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
        file: Option<Arc<super::writer::VolumeFile>>,
    ) {
        self.register_segment_inner(segment_id, volume, meta, schema, file);
    }

    fn register_segment_inner(
        &self,
        segment_id: u64,
        volume: Arc<FrozenVolume>,
        meta: SegmentMeta,
        schema: Option<&crate::core::Schema>,
        file: Option<Arc<super::writer::VolumeFile>>,
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
            let mapping = mapping_for(&manifest, &volume, seg_schema_version, schema);
            let cold = ColdSegment {
                volume,
                mapping,
                schema_version: seg_schema_version,
                visible: None,
                file,
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
            // A file-backed volume carries the handle it was opened with;
            // only a memory-backed one has to resolve a path here
            let file = volume.file_owner().or_else(|| self.file_of(segment_id));
            let cold = ColdSegment {
                mapping: super::writer::ColumnMapping {
                    sources: (0..volume.columns.len())
                        .map(super::writer::ColSource::Volume)
                        .collect(),
                    names: Vec::new(),
                    is_identity: true,
                },
                volume,
                schema_version,
                visible: None,
                file,
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
    pub fn rename(&self, new_name: &str) {
        *self.table_name.write() = SmartString::from(new_name);
        self.manifest.write().table_name = SmartString::from(new_name);
    }

    /// Renames the table and moves its files as one step under the
    /// reload lock: `move_dir` moves the directory and moves every
    /// holder's path with it, and the name changes only once the move
    /// succeeded, so a reload meanwhile waits and then looks in the
    /// right place. A failed move leaves the name and the files as they
    /// were
    pub fn rename_with(
        &self,
        new_name: &str,
        move_dir: impl FnOnce() -> std::io::Result<()>,
    ) -> std::io::Result<()> {
        let _reload = self.reloading.lock();
        move_dir()?;
        #[cfg(any(test, feature = "test-failpoints"))]
        crate::test_failpoints::rename_directory_moved();
        self.rename(new_name);
        Ok(())
    }

    /// Add tombstone row_ids with their commit_seq (when the tombstone was created).
    /// Lock order: manifest FIRST, then tombstones (matches read paths like
    /// deduped_row_count, total_row_count, check_value_exists_in_segments).
    /// The commit_seq enables snapshot isolation: older snapshots don't see
    /// newer tombstones, so the original cold row remains visible to them.
    pub fn add_tombstones(&self, row_ids: &[i64], commit_seq: u64) {
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
                    manifest.tombstones.push((rid, commit_seq));
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
                        // Update the manifest entry in-place.
                        if let Some(entry) = manifest
                            .tombstones
                            .iter_mut()
                            .find(|(r, s)| *r == rid && *s == old_seq)
                        {
                            entry.1 = commit_seq;
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
                .retain(|&(rid, _)| !row_ids.contains(&rid));
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
            manifest.tombstones.retain(|&(rid, seq)| {
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
            .entry(row_id)
            .or_insert_with(get_fast_timestamp);
    }

    /// Get pending tombstone row_ids for a transaction (for WAL recording).
    pub fn get_pending_tombstones(&self, txn_id: i64) -> Vec<i64> {
        self.pending_txn_tombstones
            .read()
            .get(&txn_id)
            .map(|set| set.keys().copied().collect())
            .unwrap_or_default()
    }

    /// Insert pending tombstones for a transaction directly into a set (no Vec clone).
    pub fn insert_pending_tombstones_into(
        &self,
        txn_id: i64,
        dest: &mut rustc_hash::FxHashSet<i64>,
    ) {
        if let Some(ids) = self.pending_txn_tombstones.read().get(&txn_id) {
            for &id in ids.keys() {
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
            .is_some_and(|set| set.contains_key(&row_id))
    }

    /// Commit pending tombstones: move from per-txn pending to shared tombstone set.
    /// The commit_seq is the transaction's commit sequence, used for snapshot
    /// isolation: older snapshots won't see these tombstones.
    pub fn commit_pending_tombstones(&self, txn_id: i64, commit_seq: u64) {
        let pending = self.pending_txn_tombstones.write().remove(&txn_id);
        if let Some(ids) = pending {
            if !ids.is_empty() {
                let id_vec: Vec<i64> = ids.into_keys().collect();
                self.add_tombstones(&id_vec, commit_seq);
            }
        }
    }

    /// Rollback pending tombstones: discard without applying.
    pub fn rollback_pending_tombstones(&self, txn_id: i64) {
        self.pending_txn_tombstones.write().remove(&txn_id);
    }

    /// Discard the pending tombstones made after a timestamp (savepoint rollback)
    /// and return the row_ids discarded, so their row claims can be released.
    pub fn rollback_pending_tombstones_after(&self, txn_id: i64, timestamp: i64) -> Vec<i64> {
        let mut discarded = Vec::new();
        if let Some(ids) = self.pending_txn_tombstones.write().get_mut(&txn_id) {
            ids.retain(|&row_id, tombstoned_at| {
                let keep = *tombstoned_at <= timestamp;
                if !keep {
                    discarded.push(row_id);
                }
                keep
            });
        }
        discarded
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
    pub fn row_exists(&self, row_id: i64) -> Result<bool> {
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return Ok(false);
        }
        // Metadata-only check (binary search on row_ids). Does not access
        // column data, so no mark_accessed — should not pin volumes.
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
                if cold.volume.locate(row_id).is_some() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Get a cold row by row_id. Returns the Row if found and not tombstoned.
    /// Iterates newest-first so overlapping row_ids return the newest version.
    /// Uses metadata-only search, reloads only the target cold volume if needed.
    pub fn get_cold_row(&self, row_id: i64) -> crate::core::Result<Option<crate::core::Row>> {
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return Ok(None);
        }
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .rev()
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
                if let Some(idx) = cold.volume.locate(row_id) {
                    if cold.volume.is_cold() {
                        if let Some(vol) = self.ensure_volume(*seg_id)? {
                            return Ok(Some(vol.get_row(idx)?));
                        }
                        // Segment removed by compaction — retry with fresh state.
                        return self.get_cold_row_retry(row_id);
                    }
                    cold.volume.mark_accessed();
                    return Ok(Some(cold.volume.get_row(idx)?));
                }
            }
        }
        Ok(None)
    }

    /// Retry get_cold_row with a fresh consistent snapshot after compaction.
    fn get_cold_row_retry(&self, row_id: i64) -> crate::core::Result<Option<crate::core::Row>> {
        self.ensure_columns();
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
                if let Some(idx) = cold.volume.locate(row_id) {
                    if cold.volume.is_cold() {
                        return Err(crate::core::Error::Internal {
                            message: format!(
                                "table '{}': cold volume reload failed for segment {}; \
                                 refusing to serve partial data",
                                self.table_name.read(),
                                seg_id
                            ),
                        });
                    }
                    cold.volume.mark_accessed();
                    return Ok(Some(cold.volume.get_row(idx)?));
                }
            }
        }
        Ok(None)
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
    ) -> crate::core::Result<Option<crate::core::Row>> {
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return Ok(None);
        }
        let (seg_ids, segments) = {
            let manifest = self.manifest.read();
            let seg_ids: Vec<(u64, i64, i64)> = manifest
                .segments
                .iter()
                .rev()
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
                if let Some(idx) = cold.volume.locate(row_id) {
                    let vol = if cold.volume.is_cold() {
                        match self.ensure_volume(*seg_id)? {
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
                    let mapping = self.get_volume_mapping(*seg_id, schema);
                    if mapping.is_identity {
                        return Ok(Some(vol.get_row(idx)?));
                    }
                    return Ok(Some(vol.get_row_mapped(idx, &mapping)?));
                }
            }
        }
        Ok(None)
    }

    /// Retry get_cold_row_normalized with a fresh consistent snapshot after compaction.
    fn get_cold_row_normalized_retry(
        &self,
        row_id: i64,
        schema: &crate::core::Schema,
    ) -> crate::core::Result<Option<crate::core::Row>> {
        self.ensure_columns();
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
                if let Some(idx) = cold.volume.locate(row_id) {
                    if cold.volume.is_cold() {
                        return Err(crate::core::Error::Internal {
                            message: format!(
                                "table '{}': cold volume reload failed for segment {}; \
                                 refusing to serve partial data",
                                self.table_name.read(),
                                seg_id
                            ),
                        });
                    }
                    cold.volume.mark_accessed();
                    let mapping = self.get_volume_mapping(*seg_id, schema);
                    if mapping.is_identity {
                        return Ok(Some(cold.volume.get_row(idx)?));
                    }
                    return Ok(Some(cold.volume.get_row_mapped(idx, &mapping)?));
                }
            }
        }
        Ok(None)
    }

    /// Check if a row_id actually exists in any loaded volume.
    /// Does NOT check tombstones. Used for idempotent WAL replay.
    /// Uses binary search on the volume's row_ids for O(log n) per segment.
    pub fn is_row_id_in_volume(&self, row_id: i64) -> Result<bool> {
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
                if cold.volume.locate(row_id).is_some() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Get the total live row count across all segments (minus tombstones).
    /// NOTE: This is a fast estimate that does not deduplicate overlapping row_ids.
    /// Use `deduped_row_count()` for an exact count.
    pub fn total_row_count(&self) -> usize {
        let manifest = self.manifest.read();
        let ts_count = self.tombstones.read().len();
        let total: usize = manifest.segments.iter().map(|s| s.row_count).sum();
        total.saturating_sub(ts_count)
    }

    /// Get the exact deduplicated row count across all segments.
    /// Uses a cached value that is invalidated on segment/tombstone changes.
    pub fn deduped_row_count(&self) -> Result<usize> {
        let cached = self
            .cached_deduped_count
            .load(std::sync::atomic::Ordering::Relaxed);
        if cached != u64::MAX {
            return Ok(cached as usize);
        }
        let count = self.compute_deduped_row_count()?;
        self.cached_deduped_count
            .store(count as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(count)
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

    /// Current generation of the segment set. Incremented when a segment is
    /// registered and when the set is cleared, so a reader that took it before
    /// sees every change that can take a row away.
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
    fn compute_deduped_row_count(&self) -> Result<usize> {
        let segments = Arc::clone(&*self.segments.read());
        if segments.is_empty() {
            return Ok(0);
        }
        let tombstones = Arc::clone(&*self.tombstones.read());
        if segments.len() == 1 {
            let mut total = 0;
            for cs in segments.values() {
                total += cs.volume.row_ids()?.len();
            }
            return Ok(total.saturating_sub(tombstones.len()));
        }

        // Fast path: use visibility bitmaps (O(1) per row, zero allocation).
        // visible=None means all rows visible (no overlap), is_visible() handles both.
        {
            let mut count = 0usize;
            for cs in segments.values() {
                let vol = &cs.volume;
                let ids = vol.row_ids()?;
                for (i, &row_id) in ids.iter().enumerate() {
                    if !cs.is_visible(i) {
                        continue;
                    }
                    if !tombstones.is_empty() && tombstones.contains_key(&row_id) {
                        continue;
                    }
                    count += 1;
                }
            }
            Ok(count)
        }
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
                names: Vec::new(),
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

    /// Record a column drop so old volumes don't leak stale data.
    /// `schema_version` is the current schema epoch at drop time. Only volumes
    /// with schema_version <= this value will have the column masked.
    pub fn record_column_drop(&self, col_name: &str, schema_version: u64, ddl_lsn: Option<u64>) {
        let lower = SmartString::from(col_name.to_lowercase());
        let mut manifest = self.manifest.write();
        // Remove any existing entry for this column name before adding the new one.
        // This handles DROP + ADD + DROP sequences correctly.
        manifest
            .dropped_columns
            .retain(|(name, _)| name.as_str() != lower.as_str());
        manifest.dropped_columns.push((lower, schema_version));
        if let Some(lsn) = ddl_lsn {
            Self::note_ddl_lsn_in(&mut manifest, lsn);
        }
    }

    /// The log position up to which this manifest holds every rename and
    /// drop, kept in the drop list under a name no column can carry; a
    /// change replayed from the log at or below it is already here
    pub fn ddl_recorded_lsn(&self) -> u64 {
        self.manifest
            .read()
            .dropped_columns
            .iter()
            .find(|(name, _)| name.as_str() == super::writer::DDL_LSN_MARKER)
            .map_or(0, |(_, lsn)| *lsn)
    }

    /// Raise the recorded log position to `lsn`
    pub fn note_ddl_lsn(&self, lsn: u64) {
        Self::note_ddl_lsn_in(&mut self.manifest.write(), lsn);
    }

    fn note_ddl_lsn_in(manifest: &mut TableManifest, lsn: u64) {
        match manifest
            .dropped_columns
            .iter_mut()
            .find(|(name, _)| name.as_str() == super::writer::DDL_LSN_MARKER)
        {
            Some((_, recorded)) => *recorded = (*recorded).max(lsn),
            None => manifest
                .dropped_columns
                .push((SmartString::from(super::writer::DDL_LSN_MARKER), lsn)),
        }
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

    /// Mark the start of an interval in which rows are leaving the table, and
    /// the end of it when the returned guard drops. A reader that took its
    /// view inside the interval cannot be answered from it: the rows it names
    /// may already be gone while the generation still reads the same.
    pub fn begin_destruction(&self) -> Destruction<'_> {
        self.destruction_count
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Destruction { mgr: self }
    }

    /// Whether a destructive publication is in flight.
    pub fn is_destruction_in_progress(&self) -> bool {
        self.destruction_count
            .load(std::sync::atomic::Ordering::Acquire)
            > 0
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
    pub fn record_column_rename(
        &self,
        old_name: &str,
        new_name: &str,
        schema_version: u64,
        ddl_lsn: Option<u64>,
    ) {
        // Persist in manifest for restart. The version goes into the drop
        // list under a name no column can carry, so the order of renames
        // against drops survives without a change of format; a reader that
        // does not know the marker never matches it against a column
        let mut manifest = self.manifest.write();
        let ordinal = manifest.column_renames.len();
        manifest
            .column_renames
            .push((SmartString::from(old_name), SmartString::from(new_name)));
        manifest.dropped_columns.push((
            SmartString::from(super::writer::rename_marker(ordinal).as_str()),
            schema_version,
        ));
        if let Some(lsn) = ddl_lsn {
            Self::note_ddl_lsn_in(&mut manifest, lsn);
        }
    }

    /// The largest schema version this table's manifest carries: of its
    /// segments, its drops and its rename markers. The engine's schema
    /// epoch restarts above it after an open, so versions recorded from
    /// then on order against the persisted ones
    pub fn max_schema_version(&self) -> u64 {
        let manifest = self.manifest.read();
        let segments = manifest.segments.iter().map(|s| s.schema_version);
        let drops = manifest
            .dropped_columns
            .iter()
            .filter(|(name, _)| name.as_str() != super::writer::DDL_LSN_MARKER)
            .map(|(_, v)| *v);
        segments.chain(drops).max().unwrap_or(0)
    }

    /// Load manifest from disk.
    pub fn load_from_disk(table_name: &str, volume_dir: &Path) -> Result<Option<Self>> {
        let table_dir = volume_dir.join(table_name);
        let manifest_path = table_dir.join("manifest.bin");

        if !manifest_path.try_exists()? {
            return Ok(None);
        }

        let manifest = TableManifest::read_from_disk(&manifest_path)?;
        let manager = Self::from_manifest(manifest, Some(volume_dir.to_path_buf()));

        Ok(Some(manager))
    }

    /// Remove all segments and tombstones (for DROP TABLE / TRUNCATE).
    pub fn clear(&self) {
        let _destruction = self.begin_destruction();
        {
            let mut manifest = self.manifest.write();
            manifest.segments.clear();
            manifest.tombstones.clear();
        }
        *self.segments.write() = Arc::new(FxHashMap::default());
        *self.tombstones.write() = Arc::new(FxHashMap::default());
        self.key_order_verdicts.lock().clear();
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        self.has_segments_flag
            .store(false, std::sync::atomic::Ordering::Relaxed);
        self.has_cold
            .store(false, std::sync::atomic::Ordering::Relaxed);
        // Clearing takes rows away as much as a seal does, so a reader that
        // took the generation before it must see the change after
        self.seal_generation
            .fetch_add(1, std::sync::atomic::Ordering::Release);
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
        self.forget_key_order(segment_ids);
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
        schema: Option<&crate::core::Schema>,
    ) {
        self.replace_segments_atomic_multi(
            vec![(new_segment_id, new_volume, new_meta)],
            old_segment_ids,
            schema,
        );
    }

    /// Atomically replace old segments with multiple new ones.
    /// Used by compaction-with-split when the merged output exceeds target_volume_rows.
    pub fn replace_segments_atomic_multi(
        &self,
        new_volumes: Vec<(u64, Arc<FrozenVolume>, SegmentMeta)>,
        old_segment_ids: &[u64],
        schema: Option<&crate::core::Schema>,
    ) {
        if new_volumes.is_empty() {
            self.replace_segments_atomic_remove_only(old_segment_ids);
            return;
        }
        self.publish_with(new_volumes, old_segment_ids, schema, |_| {});
    }

    /// The publication in its two steps, `between` run after the first
    /// and before the second; a test puts a seal there
    fn publish_with(
        &self,
        new_volumes: Vec<(u64, Arc<FrozenVolume>, SegmentMeta)>,
        old_segment_ids: &[u64],
        schema: Option<&crate::core::Schema>,
        between: impl FnOnce(&Self),
    ) -> bool {
        let owners = self.output_owners(&new_volumes);
        let prepared = self.prepare_publication(&new_volumes, old_segment_ids, &owners);
        between(self);
        let fresh =
            self.commit_publication(prepared, new_volumes, old_segment_ids, schema, &owners);
        self.forget_key_order(old_segment_ids);
        self.cached_deduped_count
            .store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        fresh
    }

    /// The handle of each output's file, taken once before the publication
    /// locks: `file_of` takes the process-wide registry lock, which a writer
    /// holding the seal fence must not wait on
    fn output_owners(
        &self,
        new_volumes: &[(u64, Arc<FrozenVolume>, SegmentMeta)],
    ) -> Vec<Option<Arc<super::writer::VolumeFile>>> {
        new_volumes
            .iter()
            .map(|(seg_id, volume, _)| volume.file_owner().or_else(|| self.file_of(*seg_id)))
            .collect()
    }

    /// The segments as they will stand after a publication, with every
    /// row's visibility decided, built outside the locks on a snapshot of
    /// the segments; `commit_publication` checks the snapshot still
    /// stands before it swaps the map in
    fn prepare_publication(
        &self,
        new_volumes: &[(u64, Arc<FrozenVolume>, SegmentMeta)],
        old_segment_ids: &[u64],
        owners: &[Option<Arc<super::writer::VolumeFile>>],
    ) -> PreparedPublication {
        let (snapshot, order, mut map) = {
            let manifest = self.manifest.read();
            let segments = self.segments.read();
            let order: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            let mut map = (**segments).clone();
            for &id in old_segment_ids {
                map.remove(&id);
            }
            for ((seg_id, vol, meta), owner) in new_volumes.iter().zip(owners) {
                map.insert(
                    *seg_id,
                    ColdSegment {
                        mapping: mapping_for(&manifest, vol, meta.schema_version, None),
                        volume: Arc::clone(vol),
                        schema_version: meta.schema_version,
                        visible: None,
                        file: owner.clone(),
                    },
                );
            }
            (Arc::clone(&*segments), order, map)
        };
        let new_order = published_order(&order, new_volumes, old_segment_ids);
        compute_visibility_bitmaps(&new_order, &mut map, &mut self.visibility_seen.lock());
        PreparedPublication {
            snapshot,
            order,
            map,
        }
    }

    /// Publishes what `prepare_publication` built when the segments are
    /// still the snapshot it was built on; otherwise builds the map again
    /// under the locks. True when the prepared map was used
    fn commit_publication(
        &self,
        prepared: PreparedPublication,
        new_volumes: Vec<(u64, Arc<FrozenVolume>, SegmentMeta)>,
        old_segment_ids: &[u64],
        schema: Option<&crate::core::Schema>,
        owners: &[Option<Arc<super::writer::VolumeFile>>],
    ) -> bool {
        let mut manifest = self.manifest.write();
        let mut segments = self.segments.write();
        let current_order: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
        let fresh = Arc::ptr_eq(&*segments, &prepared.snapshot) && current_order == prepared.order;
        let mut map = if fresh {
            prepared.map
        } else {
            let mut map = (**segments).clone();
            for &id in old_segment_ids {
                map.remove(&id);
            }
            map
        };
        let insert_pos = manifest
            .segments
            .iter()
            .position(|s| old_segment_ids.contains(&s.segment_id))
            .unwrap_or(manifest.segments.len());
        manifest.remove_segments(old_segment_ids);
        let insert_pos = insert_pos.min(manifest.segments.len());
        for (i, (seg_id, vol, meta)) in new_volumes.into_iter().enumerate() {
            let owner = owners[i].clone();
            if seg_id >= manifest.next_segment_id {
                manifest.next_segment_id = seg_id + 1;
            }
            let seg_schema_version = meta.schema_version;
            manifest.segments.insert(insert_pos + i, meta);
            // The mapping reads the manifest's history as it is now
            let mapping = mapping_for(&manifest, &vol, seg_schema_version, schema);
            match map.get_mut(&seg_id) {
                Some(cs) if fresh => cs.mapping = mapping,
                _ => {
                    map.insert(
                        seg_id,
                        ColdSegment {
                            mapping,
                            volume: vol,
                            schema_version: seg_schema_version,
                            visible: None,
                            file: owner,
                        },
                    );
                }
            }
        }
        if !fresh {
            let seg_ids: Vec<u64> = manifest.segments.iter().map(|m| m.segment_id).collect();
            compute_visibility_bitmaps(&seg_ids, &mut map, &mut self.visibility_seen.lock());
        }
        if self.has_cold.load(std::sync::atomic::Ordering::Relaxed)
            && !map.values().any(|cs| cs.volume.is_cold())
        {
            self.has_cold
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        // Whether the table has segments is read from the map published,
        // under the locks that publish it
        self.has_segments_flag
            .store(!map.is_empty(), std::sync::atomic::Ordering::Relaxed);
        *segments = Arc::new(map);
        fresh
    }

    /// Atomically remove old segments without adding a replacement.
    /// Used when partial compaction finds all rows in merged volumes are tombstoned.
    pub fn replace_segments_atomic_remove_only(&self, old_segment_ids: &[u64]) {
        self.publish_with(Vec::new(), old_segment_ids, None, |_| {});
    }

    /// Get the manifest for reading (e.g., to iterate segment metadata).
    pub fn manifest(&self) -> parking_lot::RwLockReadGuard<'_, TableManifest> {
        self.manifest.read()
    }

    /// CoW snapshot of the loaded segments map.
    /// Reloads cold volumes first so column data is available.
    /// Does NOT mark volumes as accessed — callers that need eviction
    /// protection should mark the specific volumes they use.
    pub fn segments_snapshot(&self) -> crate::core::Result<Arc<FxHashMap<u64, ColdSegment>>> {
        self.ensure_columns();
        let segs = Arc::clone(&*self.segments.read());
        if !segs.values().any(|cs| cs.volume.is_cold()) {
            return Ok(segs);
        }
        // Race with eviction: retry once, then fail closed. Filtering the
        // cold segments out would hand callers a silently smaller table.
        self.ensure_columns();
        let segs = Arc::clone(&*self.segments.read());
        let cold: Vec<u64> = segs
            .iter()
            .filter(|(_, cs)| cs.volume.is_cold())
            .map(|(&id, _)| id)
            .collect();
        if !cold.is_empty() {
            return Err(crate::core::Error::Internal {
                message: format!(
                    "table '{}': cold volume reload failed for segment(s) {:?}; \
                     refusing to serve partial data",
                    self.table_name.read(),
                    cold
                ),
            });
        }
        Ok(segs)
    }

    /// Raw CoW snapshot without ensure_columns. Callers that only need
    /// metadata (row_ids, zone maps) use this to avoid reloading cold volumes.
    pub fn segments_raw(&self) -> Arc<FxHashMap<u64, ColdSegment>> {
        Arc::clone(&*self.segments.read())
    }

    /// Reload a single cold volume by segment ID. Returns the loaded volume
    /// if successful. Used by point lookups to avoid reloading all cold volumes.
    /// `Ok(None)` means the segment was removed by concurrent compaction
    /// (the caller may retry with the current manifest or skip it);
    /// `Err` means the cold volume could not be reloaded and the caller
    /// must fail closed rather than serve partial data.
    pub fn ensure_volume(&self, seg_id: u64) -> crate::core::Result<Option<Arc<FrozenVolume>>> {
        self.ensure_volume_inner(seg_id, None)
    }

    /// The same reload for a reader that pinned the segment's file. The
    /// reader's handle is what reads it, so a segment the manifest has since
    /// dropped still loads; that segment is not put back in the map, and the
    /// next plain reload still finds it gone.
    pub fn ensure_pinned_volume(
        &self,
        seg_id: u64,
        handle: &Arc<super::writer::VolumeFile>,
    ) -> crate::core::Result<Option<Arc<FrozenVolume>>> {
        self.ensure_volume_inner(seg_id, Some(handle))
    }

    /// The volume's bytes: through the reader's handle when it has one, and
    /// from the segment's own path otherwise.
    fn read_volume_for(
        &self,
        seg_id: u64,
        pinned: Option<&Arc<super::writer::VolumeFile>>,
    ) -> crate::core::Result<Arc<FrozenVolume>> {
        let volume = match pinned {
            Some(handle) => super::io::read_volume_from_handle(handle)?,
            None => {
                let vol_dir =
                    self.volume_dir
                        .as_ref()
                        .ok_or_else(|| crate::core::Error::Internal {
                            message: format!(
                            "table '{}': cold segment {} has no volume directory to reload from",
                            self.table_name.read(),
                            seg_id
                        ),
                        })?;
                let filename = format!("vol_{:016x}.vol", seg_id);
                let full_path = vol_dir.join(self.table_name.read().as_str()).join(filename);
                super::io::read_volume_from_disk(&full_path)?
            }
        };
        Ok(Arc::new(volume))
    }

    fn reload_error(&self, seg_id: u64, e: crate::core::Error) -> crate::core::Error {
        crate::core::Error::Internal {
            message: format!(
                "table '{}': failed to reload cold volume seg={}: {}; \
                 refusing to serve partial data",
                self.table_name.read(),
                seg_id,
                e
            ),
        }
    }

    fn ensure_volume_inner(
        &self,
        seg_id: u64,
        pinned: Option<&Arc<super::writer::VolumeFile>>,
    ) -> crate::core::Result<Option<Arc<FrozenVolume>>> {
        // A segment the manifest has dropped is still readable through a
        // pinned handle, and stays out of the map
        let load_dropped = |this: &Self| -> crate::core::Result<Option<Arc<FrozenVolume>>> {
            let Some(handle) = pinned else {
                return Ok(None);
            };
            let volume = this
                .read_volume_for(seg_id, Some(handle))
                .map_err(|e| this.reload_error(seg_id, e))?;
            volume.mark_accessed();
            Ok(Some(volume))
        };

        // Where the segment stands, read without a guard outliving the look:
        // a file read under the segments lock would hold it across every
        // block of the volume
        enum Stands {
            Loaded(Arc<FrozenVolume>),
            Cold,
            Gone,
        }
        let look = |this: &Self| -> Stands {
            let segs = this.segments.read();
            match segs.get(&seg_id) {
                None => Stands::Gone,
                Some(cs) if !cs.volume.is_cold() => {
                    cs.volume.mark_accessed();
                    Stands::Loaded(Arc::clone(&cs.volume))
                }
                Some(_) => Stands::Cold,
            }
        };

        // Fast path: already loaded (or another thread just loaded it)
        match look(self) {
            Stands::Loaded(volume) => return Ok(Some(volume)),
            Stands::Gone => return load_dropped(self),
            Stands::Cold => {}
        }
        // Serialize reloads to prevent a concurrent stampede on one volume.
        // Second thread re-checks the fast path after acquiring the guard.
        let _guard = self.reloading.lock();
        match look(self) {
            Stands::Loaded(volume) => return Ok(Some(volume)),
            // The reload of a segment no longer in the map touches nothing
            // the guard protects, so it goes without it
            Stands::Gone => {
                drop(_guard);
                return load_dropped(self);
            }
            Stands::Cold => {}
        }
        let volume = self
            .read_volume_for(seg_id, pinned)
            .map_err(|e| self.reload_error(seg_id, e))?;
        volume.mark_accessed();
        let mut segments = self.segments.write();
        let mut new_map = (**segments).clone();
        // A segment that left the manifest while the file was read: a pinned
        // reader keeps what it read, and without one the caller is told, since
        // the rows live in a newer volume now
        let cs = match new_map.get_mut(&seg_id) {
            None if pinned.is_some() => return Ok(Some(volume)),
            None => return Ok(None),
            Some(cs) => cs,
        };
        if !cs.volume.unique_indices.read().is_empty() {
            *volume.unique_indices.write() = std::mem::take(&mut *cs.volume.unique_indices.write());
        }
        volume.inherit_row_order(&cs.volume);
        // Put it back for everyone: the entry keeps its mapping and its
        // visibility bitmap, and the next read shares this volume instead of
        // reading the file again
        cs.volume = Arc::clone(&volume);
        let still_cold = new_map.values().any(|cs| cs.volume.is_cold());
        *segments = Arc::new(new_map);
        if !still_cold {
            self.has_cold
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(Some(volume))
    }

    /// Get the manifest for writing (e.g., to allocate segment IDs).
    pub fn manifest_mut(&self) -> parking_lot::RwLockWriteGuard<'_, TableManifest> {
        self.manifest.write()
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
        });

        assert_eq!(m.find_segment_for_row_id(50).unwrap().1.segment_id, 1);
        assert_eq!(m.find_segment_for_row_id(150).unwrap().1.segment_id, 2);
        assert!(m.find_segment_for_row_id(300).is_none());
    }

    #[test]
    fn test_manifest_serialize_roundtrip() {
        let mut m = TableManifest::new("my_table");
        m.next_segment_id = 5;
        m.checkpoint_lsn = 42;
        m.tombstones = vec![(10, 0), (20, 0), (30, 0)];
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("seg_0001.vol"),
            row_count: 10000,
            min_row_id: 1,
            max_row_id: 10000,
            creation_lsn: 10,
            seal_seq: 0,
            schema_version: 3,
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
        assert_eq!(loaded.tombstones, vec![(10, 0), (20, 0), (30, 0)]);
    }

    #[test]
    fn test_manifest_disk_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("manifest.bin");

        let mut m = TableManifest::new("disk_test");
        m.tombstones = vec![(5, 0), (10, 0)];
        m.add_segment(SegmentMeta {
            segment_id: 1,
            file_path: PathBuf::from("vol.vol"),
            row_count: 100,
            min_row_id: 1,
            max_row_id: 100,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        });

        m.write_to_disk(&path).unwrap();
        let loaded = TableManifest::read_from_disk(&path).unwrap();

        assert_eq!(loaded.table_name.as_str(), "disk_test");
        assert_eq!(loaded.segments.len(), 1);
        assert_eq!(loaded.tombstones, vec![(5, 0), (10, 0)]);
    }

    /// The guard marks the whole interval in which rows are leaving, so a
    /// reader that consults it is told so until the outermost one drops.
    #[test]
    fn test_destruction_guard_marks_the_interval() {
        let mgr = SegmentManager::new("test", None);
        assert!(!mgr.is_destruction_in_progress());
        {
            let _outer = mgr.begin_destruction();
            assert!(mgr.is_destruction_in_progress());
            {
                let _inner = mgr.begin_destruction();
                assert!(mgr.is_destruction_in_progress());
            }
            assert!(
                mgr.is_destruction_in_progress(),
                "the outer guard still holds the interval open"
            );
        }
        assert!(!mgr.is_destruction_in_progress());
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
        let volume = Arc::new(builder.finish().unwrap());

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
        };
        mgr.register_segment(1, volume, meta, None, None);

        assert_eq!(mgr.segment_count(), 1);
        assert_eq!(mgr.total_row_count(), 10);
        assert!(mgr.row_exists(5).unwrap());
        assert!(!mgr.row_exists(11).unwrap());
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
        let volume = Arc::new(builder.finish().unwrap());

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
            },
            None,
            None,
        );

        // Tombstone row_id=5 (commit_seq=1)
        mgr.add_tombstones(&[5], 1);
        assert!(!mgr.row_exists(5).unwrap());
        assert!(mgr.row_exists(4).unwrap());
        assert!(mgr.row_exists(6).unwrap());
        assert_eq!(mgr.total_row_count(), 9);
        assert!(mgr.is_tombstoned(5));
        assert!(!mgr.is_tombstoned(4));

        // Clear tombstones
        mgr.clear_tombstones();
        assert!(mgr.row_exists(5).unwrap());
        assert_eq!(mgr.total_row_count(), 10);
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
            let vol = Arc::new(builder.finish().unwrap());
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
                },
                None,
                None,
            );
        }

        let newest_first = mgr.get_volumes_newest_first().unwrap();
        assert_eq!(newest_first.len(), 3);
        // get_volumes_newest_first reverses manifest insertion order.
        // Segments were registered as [1, 3, 2], so reversed = [2, 3, 1].
        assert_eq!(newest_first[0].0, 2);
        assert_eq!(newest_first[1].0, 3);
        assert_eq!(newest_first[2].0, 1);
    }

    #[test]
    fn a_recluster_request_opened_during_a_check_stays_open() {
        let mgr = SegmentManager::new("t", None);
        let first = mgr
            .recluster_request()
            .expect("a new manager starts with a request open");
        // The key changes while the check for `first` runs
        mgr.request_recluster();
        mgr.recluster_done(first);
        assert!(
            mgr.recluster_pending(),
            "closing the older request closed the one opened since"
        );
        let second = mgr.recluster_request().unwrap();
        mgr.recluster_done(second);
        assert!(!mgr.recluster_pending());
        // A stale close changes nothing
        mgr.recluster_done(first);
        assert!(!mgr.recluster_pending());
    }

    /// One row per id, `k` walking down as the id walks up, so the
    /// volume is in id order and out of `k` order
    fn descending_k_volume(schema: &crate::core::Schema, rows: i64) -> Arc<FrozenVolume> {
        use crate::core::{Row, Value};
        let mut builder = super::super::writer::VolumeBuilder::new(schema);
        for i in 1..=rows {
            builder.add_row(
                i,
                &Row::from_values(vec![
                    Value::Integer(i),
                    Value::Integer(i),
                    Value::Integer(rows - i),
                ]),
            );
        }
        Arc::new(builder.finish().unwrap())
    }

    fn meta_of(seg_id: u64, rows: i64) -> SegmentMeta {
        SegmentMeta {
            segment_id: seg_id,
            file_path: PathBuf::from(format!("vol_{}.vol", seg_id)),
            row_count: rows as usize,
            min_row_id: 1,
            max_row_id: rows,
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        }
    }

    #[test]
    fn a_key_is_checked_through_the_volume_column_mapping() {
        use crate::core::{DataType, SchemaBuilder};
        let sealed_with = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("a", DataType::Integer, false, false)
            .column("k", DataType::Integer, false, false)
            .build();
        // `a` was dropped since: k is schema column 1 and volume column 2
        let current = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("k", DataType::Integer, false, false)
            .build();
        let mgr = SegmentManager::new("t", None);
        mgr.register_segment(
            1,
            descending_k_volume(&sealed_with, 8),
            meta_of(1, 8),
            Some(&current),
            None,
        );
        assert_eq!(mgr.known_key_order(1, &[1]), None);
        assert!(
            !mgr.decide_key_order(1, &[1]).unwrap(),
            "the verdict read volume column 1 (ascending a) for schema column 1 (descending k)"
        );
        assert_eq!(mgr.known_key_order(1, &[1]), Some(false));
        // A key column the volume predates orders nothing
        let widened = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("k", DataType::Integer, false, false)
            .column("z", DataType::Integer, true, false)
            .build();
        mgr.invalidate_mappings(&widened);
        assert!(mgr.decide_key_order(1, &[2]).unwrap());
        assert!(!mgr.decide_key_order(1, &[2, 1]).unwrap());
    }

    #[test]
    fn a_recorded_order_is_kept_by_physical_column_and_read_through_the_mapping() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};
        // Built and sealed as (id, d, k, v), k the key at position 2 and in
        // order; d is dropped afterwards, so the schema's k is at 1 and v at 2
        let sealed_with = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("d", DataType::Integer, false, false)
            .column("k", DataType::Integer, false, false)
            .column("v", DataType::Integer, false, false)
            .build();
        let mut builder = super::super::writer::VolumeBuilder::new(&sealed_with);
        for i in 1..=4i64 {
            builder.add_row(
                i,
                &Row::from_values(vec![
                    Value::Integer(i),
                    Value::Integer(0),
                    Value::Integer(i),
                    Value::Integer(5 - i),
                ]),
            );
        }
        let volume = Arc::new(builder.finish().unwrap());
        let mgr = SegmentManager::new("t", None);
        mgr.register_segment(1, volume, meta_of(1, 4), Some(&sealed_with), None);
        // The drop completes before the writer records the order it wrote:
        // the schema's positions have moved, the volume's have not
        let current = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("k", DataType::Integer, false, false)
            .column("v", DataType::Integer, false, false)
            .build();
        mgr.record_column_drop("d", 1, None);
        mgr.invalidate_mappings(&current);
        mgr.record_key_order(1, &[2]);
        assert_eq!(
            mgr.known_key_order(1, &[1]),
            Some(true),
            "k moved to position 1 and is still the recorded ordered column"
        );
        assert_eq!(
            mgr.known_key_order(1, &[2]),
            None,
            "v at position 2 was never checked; a record by the old key position would claim it"
        );
        assert!(!mgr.decide_key_order(1, &[2]).unwrap());
    }

    #[test]
    fn a_verdict_leaves_with_its_segment() {
        use crate::core::{DataType, SchemaBuilder};
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .column("a", DataType::Integer, false, false)
            .column("k", DataType::Integer, false, false)
            .build();
        let mgr = SegmentManager::new("t", None);
        for seg_id in [1u64, 2, 3, 4] {
            mgr.register_segment(
                seg_id,
                descending_k_volume(&schema, 4),
                meta_of(seg_id, 4),
                Some(&schema),
                None,
            );
            assert!(!mgr.decide_key_order(seg_id, &[2]).unwrap());
        }
        mgr.replace_segments_atomic(
            5,
            descending_k_volume(&schema, 4),
            meta_of(5, 4),
            &[1],
            None,
        );
        mgr.remove_segments(&[2]);
        mgr.replace_segments_atomic_remove_only(&[3]);
        mgr.record_key_order(5, &[2]);
        {
            let verdicts = mgr.key_order_verdicts.lock();
            assert!(
                !verdicts.contains_key(&1)
                    && !verdicts.contains_key(&2)
                    && !verdicts.contains_key(&3)
            );
            assert_eq!(verdicts.get(&4).map(|v| v.1), Some(false));
            assert_eq!(verdicts.get(&5).map(|v| v.1), Some(true));
        }
        mgr.clear();
        assert!(mgr.key_order_verdicts.lock().is_empty());
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
        });
        mgr.add_tombstones(&[5], 1);

        assert_eq!(mgr.segment_count(), 1);
        mgr.clear();
        assert_eq!(mgr.segment_count(), 0);
        assert!(mgr.tombstone_set_arc().is_empty());
    }

    static EVICTION_EPOCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A hot volume of `ids` with a compressed store attached
    fn volume_of(ids: &[i64]) -> Arc<FrozenVolume> {
        use crate::core::{DataType, Row, SchemaBuilder, Value};
        let schema = SchemaBuilder::new("t")
            .column("id", DataType::Integer, false, true)
            .build();
        let mut builder = super::super::writer::VolumeBuilder::new(&schema);
        builder.allow_any_row_order();
        for &id in ids {
            builder.add_row(id, &Row::from_values(vec![Value::Integer(id)]));
        }
        let mut volume = builder.finish().unwrap();
        let (_, store) = crate::storage::volume::io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        Arc::new(volume)
    }

    fn meta_for_ids(id: u64, ids: &[i64]) -> SegmentMeta {
        SegmentMeta {
            segment_id: id,
            file_path: PathBuf::from(format!("{id}.vol")),
            row_count: ids.len(),
            min_row_id: *ids.iter().min().unwrap_or(&0),
            max_row_id: *ids.iter().max().unwrap_or(&0),
            creation_lsn: 0,
            seal_seq: 0,
            schema_version: 0,
        }
    }

    /// The visibility every segment would get from a computation over the
    /// manager's segments as they stand
    fn reference_visibility(mgr: &SegmentManager) -> Vec<(u64, Option<Vec<u64>>)> {
        let order: Vec<u64> = mgr
            .manifest
            .read()
            .segments
            .iter()
            .map(|m| m.segment_id)
            .collect();
        let mut map = (*mgr.segments_raw()).clone();
        let mut seen = FxHashSet::default();
        compute_visibility_bitmaps(&order, &mut map, &mut seen);
        order
            .iter()
            .map(|id| (*id, map[id].visible.as_ref().map(|v| (**v).clone())))
            .collect()
    }

    fn published_visibility(mgr: &SegmentManager) -> Vec<(u64, Option<Vec<u64>>)> {
        let order: Vec<u64> = mgr
            .manifest
            .read()
            .segments
            .iter()
            .map(|m| m.segment_id)
            .collect();
        let map = mgr.segments_raw();
        order
            .iter()
            .map(|id| (*id, map[id].visible.as_ref().map(|v| (**v).clone())))
            .collect()
    }

    #[test]
    fn a_publication_prepared_outside_the_locks_is_the_one_computed_under_them() {
        let mgr = SegmentManager::new("publish", None);
        let a: Vec<i64> = (1..=70).collect();
        let b: Vec<i64> = (50..=120).collect();
        mgr.register_segment(1, volume_of(&a), meta_for_ids(1, &a), None, None);
        mgr.register_segment(2, volume_of(&b), meta_for_ids(2, &b), None, None);
        // A is rewritten as C; C keeps A's place, and B still masks the ids
        // it shares with C
        let c: Vec<i64> = (1..=70).collect();
        let outputs = vec![(3, volume_of(&c), meta_for_ids(3, &c))];
        let owners = mgr.output_owners(&outputs);
        let prepared = mgr.prepare_publication(&outputs, &[1], &owners);
        assert!(mgr.commit_publication(prepared, outputs, &[1], None, &owners));
        let order: Vec<u64> = mgr
            .manifest
            .read()
            .segments
            .iter()
            .map(|m| m.segment_id)
            .collect();
        assert_eq!(order, vec![3, 2]);
        assert_eq!(published_visibility(&mgr), reference_visibility(&mgr));
        let c_visible = mgr.segments_raw()[&3]
            .visible
            .clone()
            .expect("C overlaps B");
        assert!(c_visible[0] & 1 == 1, "id 1 is C's alone");
        let masked = (0..70)
            .filter(|i| c_visible[i >> 6] & (1u64 << (i & 63)) == 0)
            .count();
        assert_eq!(masked, 21, "ids 50 to 70 are B's");
    }

    #[test]
    fn a_publication_prepared_on_a_stale_snapshot_is_computed_under_the_locks() {
        let mgr = SegmentManager::new("publish_stale", None);
        let a: Vec<i64> = (1..=70).collect();
        let b: Vec<i64> = (50..=120).collect();
        mgr.register_segment(1, volume_of(&a), meta_for_ids(1, &a), None, None);
        mgr.register_segment(2, volume_of(&b), meta_for_ids(2, &b), None, None);
        let c: Vec<i64> = (1..=70).collect();
        let outputs = vec![(3, volume_of(&c), meta_for_ids(3, &c))];
        let owners = mgr.output_owners(&outputs);
        let prepared = mgr.prepare_publication(&outputs, &[1], &owners);
        // A seal lands in between: D takes ids 2 and 130 from everyone below
        let d: Vec<i64> = vec![2, 130];
        mgr.register_segment(4, volume_of(&d), meta_for_ids(4, &d), None, None);
        assert!(!mgr.commit_publication(prepared, outputs, &[1], None, &owners));
        let order: Vec<u64> = mgr
            .manifest
            .read()
            .segments
            .iter()
            .map(|m| m.segment_id)
            .collect();
        assert_eq!(order, vec![3, 2, 4]);
        assert_eq!(published_visibility(&mgr), reference_visibility(&mgr));
        let c_visible = mgr.segments_raw()[&3]
            .visible
            .clone()
            .expect("C overlaps B and D");
        assert!(c_visible[0] & (1 << 1) == 0, "id 2 is D's");
        assert!(c_visible[0] & 1 == 1, "id 1 is C's");
    }

    #[test]
    fn a_publication_of_nothing_keeps_the_volume_a_seal_added_meanwhile_visible() {
        let mgr = SegmentManager::new("publish_empty", None);
        let a: Vec<i64> = (1..=10).collect();
        mgr.register_segment(1, volume_of(&a), meta_for_ids(1, &a), None, None);
        // Every row of A is gone: the compaction publishes nothing, and a
        // seal lands between its two steps
        let b: Vec<i64> = vec![20, 21];
        let fresh = mgr.publish_with(Vec::new(), &[1], None, |mgr| {
            mgr.register_segment(2, volume_of(&b), meta_for_ids(2, &b), None, None);
        });
        assert!(!fresh);
        assert!(mgr.has_segments(), "the sealed volume is there to read");
        let order: Vec<u64> = mgr
            .manifest
            .read()
            .segments
            .iter()
            .map(|m| m.segment_id)
            .collect();
        assert_eq!(order, vec![2]);
        assert!(mgr.segments_raw().contains_key(&2));
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
        let mut volume = builder.finish().unwrap();
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
            },
            None,
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
            let vols = mgr.get_volumes_newest_first().unwrap();
            let (_, cs) = &vols[0];
            let row = cs.volume.get_row(0).unwrap();
            assert_eq!(row[0], Value::Integer(1));
        }

        // GLOBAL_EVICTION_EPOCH is process-global: epoch-driving tests
        // must serialize and use base-relative epochs or they corrupt
        // each other's idle-cycle accounting under parallel test runs.
        let _epoch_guard = EVICTION_EPOCH_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let base =
            super::super::writer::GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed);

        // Helper: mirror engine's evict_idle_volumes behavior.
        // Engine does: fetch_add → GLOBAL.fetch_max → mgr.evict_idle_volumes.
        let run_eviction = |mgr: &SegmentManager, epoch: u64| {
            super::super::writer::GLOBAL_EVICTION_EPOCH
                .fetch_max(base + epoch, std::sync::atomic::Ordering::Relaxed);
            mgr.evict_idle_volumes(base + epoch);
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
            let vols = mgr.get_volumes_newest_first().unwrap();
            let (_, cs) = &vols[0];
            let row = cs.volume.get_row(49).unwrap();
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
        // The volume was marked u64::MAX by get_volumes_newest_first and
        // the eviction at epoch 4 reset it to 4. The row read above went
        // through the row's group, not the whole column, so the volume is
        // still warm: at epoch 6 the delta is 2 and it stays, at epoch 7
        // the delta is 3 and warm goes cold
        {
            let segs = mgr.segments_raw();
            let cs = segs.get(&1).unwrap();
            assert!(cs.volume.is_warm(), "epoch 6: delta=2, still warm");
        }
        run_eviction(&mgr, 7);
        {
            let segs = mgr.segments_raw();
            let cs = segs.get(&1).unwrap();
            assert!(
                cs.volume.is_cold(),
                "epoch 7: should be cold after 3 idle cycles"
            );
            assert!(
                !cs.volume.columns.has_compressed_store(),
                "cold has no compressed store"
            );
        }

        // Metadata still accessible on cold volumes.
        assert!(
            mgr.row_exists(50).unwrap(),
            "metadata (row_ids) should work on cold"
        );
        assert_eq!(mgr.total_row_count(), 100);

        // Volume is still in the map (not removed).
        assert_eq!(mgr.segment_count(), 1);
    }

    #[test]
    fn a_rename_moves_the_files_before_the_name_changes() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};
        let dir = tempfile::tempdir().unwrap();
        let schema = SchemaBuilder::new("before")
            .column("id", DataType::Integer, false, true)
            .build();
        let mut builder = super::super::writer::VolumeBuilder::new(&schema);
        for i in 1..=10i64 {
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
        }
        let volume = builder.finish().unwrap();
        let path =
            crate::storage::volume::io::write_volume_to_disk(dir.path(), "before", 1, &volume)
                .unwrap();
        let mgr = SegmentManager::new("before", Some(dir.path().to_path_buf()));
        mgr.register_segment(
            1,
            Arc::new(crate::storage::volume::io::read_volume_from_disk(&path).unwrap()),
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("vol_0000000000000001.vol"),
                row_count: 10,
                min_row_id: 1,
                max_row_id: 10,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            None,
            None,
        );
        assert!(mgr.segments_raw().get(&1).unwrap().volume.is_warm());
        // A reader holds the volume across the move; while the files move
        // the name is still the old one
        let held = Arc::clone(&mgr.segments_raw().get(&1).unwrap().volume);
        mgr.rename_with("after", || {
            assert_eq!(mgr.table_name(), "before");
            super::super::writer::VolumeFile::relocate(
                &dir.path().join("before"),
                &dir.path().join("after"),
                || std::fs::rename(dir.path().join("before"), dir.path().join("after")),
            )
        })
        .unwrap();
        assert_eq!(mgr.table_name(), "after");
        // The reader still holding the volume reads it at its new place,
        // and a reload finds it there too
        assert_eq!(held.get_row(9).unwrap()[0], Value::Integer(10));
        let cold = Arc::new(held.to_cold());
        drop(held);
        drop(cold);
        let reloaded = mgr.ensure_volume(1).unwrap().unwrap();
        assert_eq!(reloaded.get_row(9).unwrap()[0], Value::Integer(10));
        // A failed move leaves the name as it was
        let error = mgr
            .rename_with("elsewhere", || Err(std::io::Error::other("no move")))
            .unwrap_err();
        assert_eq!(error.to_string(), "no move");
        assert_eq!(mgr.table_name(), "after");
    }

    #[test]
    fn test_cold_reload_failure_fails_closed() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("cold_fail")
            .column("id", DataType::Integer, false, true)
            .build();

        // No volume_dir: once the volume goes cold there is nowhere to
        // reload it from, which models a reload failure deterministically.
        let mgr = SegmentManager::new("cold_fail", None);
        let mut builder = super::super::writer::VolumeBuilder::new(&schema);
        for i in 1..=100i64 {
            builder.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
        }
        let mut volume = builder.finish().unwrap();
        let (_, store) = crate::storage::volume::io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        mgr.register_segment(
            1,
            Arc::new(volume),
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("cold_fail.vol"),
                row_count: 100,
                min_row_id: 1,
                max_row_id: 100,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            None,
            None,
        );

        let _epoch_guard = EVICTION_EPOCH_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let base =
            super::super::writer::GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        let run_eviction = |mgr: &SegmentManager, epoch: u64| {
            super::super::writer::GLOBAL_EVICTION_EPOCH
                .fetch_max(base + epoch, std::sync::atomic::Ordering::Relaxed);
            mgr.evict_idle_volumes(base + epoch);
        };
        for epoch in 0..=10u64 {
            run_eviction(&mgr, epoch);
        }
        assert!(
            mgr.segments_raw().get(&1).unwrap().volume.is_cold(),
            "precondition: volume must be cold"
        );

        // Every data-access path must fail closed instead of silently
        // serving a partial (or empty) view of the table.
        assert!(mgr.ensure_volume(1).is_err(), "ensure_volume must fail");
        assert!(
            mgr.get_volumes_newest_first().is_err(),
            "get_volumes_newest_first must fail instead of dropping the segment"
        );
        assert!(mgr.get_cold_row(50).is_err(), "get_cold_row must fail");
        assert!(
            mgr.segments_snapshot().is_err(),
            "the DML statement snapshot must fail before any mutation"
        );
        assert!(
            mgr.statement_snapshot().is_err(),
            "statement_snapshot must fail closed too"
        );
    }

    #[test]
    fn test_statement_snapshot_survives_concurrent_compaction() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("snap_compact")
            .column("id", DataType::Integer, false, true)
            .build();
        let mgr = SegmentManager::new("snap_compact", None);

        let build_volume = |vals: std::ops::RangeInclusive<i64>| {
            let mut b = super::super::writer::VolumeBuilder::new(&schema);
            for i in vals {
                b.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
            }
            Arc::new(b.finish().unwrap())
        };

        mgr.register_segment(
            1,
            build_volume(1..=10),
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("s1.vol"),
                row_count: 10,
                min_row_id: 1,
                max_row_id: 10,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            None,
            None,
        );

        let snap = mgr.statement_snapshot().unwrap();

        // Background compaction replaces segment 1 with segment 2 while the
        // statement snapshot is live.
        mgr.replace_segments_atomic(
            2,
            build_volume(1..=10),
            SegmentMeta {
                segment_id: 2,
                file_path: PathBuf::from("s2.vol"),
                row_count: 10,
                min_row_id: 1,
                max_row_id: 10,
                creation_lsn: 0,
                seal_seq: 1,
                schema_version: 0,
            },
            &[1],
            None,
        );

        // The snapshot must keep serving ITS view: segment 1, one volume.
        // (The live-manifest ordering bug returned zero volumes here.)
        let vols = snap.volumes_newest_first();
        assert_eq!(vols.len(), 1, "snapshot must not lose its segment");
        assert_eq!(vols[0].0, 1);
        assert_eq!(vols[0].1.volume.get_row(0).unwrap()[0], Value::Integer(1));
    }
    #[test]
    fn test_unique_lookup_uses_statement_snapshot_not_live_state() {
        use crate::core::{DataType, Row, SchemaBuilder, Value};

        let schema = SchemaBuilder::new("uniq_snap")
            .column("id", DataType::Integer, false, true)
            .build();
        // No volume_dir: once the live map goes cold, live lookups must
        // fail to reload while the statement snapshot keeps working.
        let mgr = SegmentManager::new("uniq_snap", None);
        let mut b = super::super::writer::VolumeBuilder::new(&schema);
        for i in 1..=10i64 {
            b.add_row(i, &Row::from_values(vec![Value::Integer(i)]));
        }
        let mut volume = b.finish().unwrap();
        let (_, store) = crate::storage::volume::io::serialize_v4_public(&volume).unwrap();
        volume.columns.attach_compressed_store(store);
        mgr.register_segment(
            1,
            Arc::new(volume),
            SegmentMeta {
                segment_id: 1,
                file_path: PathBuf::from("uniq.vol"),
                row_count: 10,
                min_row_id: 1,
                max_row_id: 10,
                creation_lsn: 0,
                seal_seq: 0,
                schema_version: 0,
            },
            None,
            None,
        );

        let snap = mgr.statement_snapshot().unwrap();

        let _epoch_guard = EVICTION_EPOCH_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let base =
            super::super::writer::GLOBAL_EVICTION_EPOCH.load(std::sync::atomic::Ordering::Relaxed);
        for e in 0..=10u64 {
            super::super::writer::GLOBAL_EVICTION_EPOCH
                .fetch_max(base + e, std::sync::atomic::Ordering::Relaxed);
            mgr.evict_idle_volumes(base + e);
        }
        assert!(
            mgr.segments_raw().get(&1).unwrap().volume.is_cold(),
            "precondition: live map must be cold"
        );

        let target = Value::Integer(5);
        let defaults = [Value::Integer(0)];
        // Review probe: the live lookup fails to reload...
        assert!(
            mgr.find_row_id_by_values(&[0], &[&target], &defaults)
                .is_err(),
            "live lookup must fail on unreloadable cold state"
        );
        // ...while the statement-snapshot lookup keeps serving its view.
        let found = mgr
            .find_row_id_by_values_in(
                &snap.seg_ids_newest_first,
                &snap.segs,
                &snap.tombstones,
                &[0],
                &[&target],
                &defaults,
            )
            .unwrap();
        assert_eq!(found, Some(5), "snapshot lookup must succeed");
    }
}
