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
    /// Compaction epoch when this segment was last written/merged.
    /// Used to force-include stale volumes after max_age cycles.
    pub compaction_epoch: u64,
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

    /// Serialize the manifest to bytes (V2 format with tombstones).
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
            buf.write_all(&seg.compaction_epoch.to_le_bytes())?;
            buf.write_all(&seg.schema_version.to_le_bytes())?;

            // File path as UTF-8 string
            let path_str = seg.file_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            buf.write_all(&(path_bytes.len() as u32).to_le_bytes())?;
            buf.write_all(path_bytes)?;
        }

        // Tombstones (V4: row_id + commit_seq pairs; V2-V3: row_id only)
        buf.write_all(&(self.tombstones.len() as u64).to_le_bytes())?;
        for &(row_id, commit_seq) in &self.tombstones {
            buf.write_all(&row_id.to_le_bytes())?;
            buf.write_all(&commit_seq.to_le_bytes())?;
        }

        // Column renames (V5+): (old_name, new_name) pairs
        buf.write_all(&(self.column_renames.len() as u32).to_le_bytes())?;
        for (old_name, new_name) in &self.column_renames {
            let ob = old_name.as_bytes();
            buf.write_all(&(ob.len() as u16).to_le_bytes())?;
            buf.write_all(ob)?;
            let nb = new_name.as_bytes();
            buf.write_all(&(nb.len() as u16).to_le_bytes())?;
            buf.write_all(nb)?;
        }

        // Dropped columns (V6+: name + schema_version; V5: name only)
        buf.write_all(&(self.dropped_columns.len() as u32).to_le_bytes())?;
        for (name, version) in &self.dropped_columns {
            let nb = name.as_bytes();
            buf.write_all(&(nb.len() as u16).to_le_bytes())?;
            buf.write_all(nb)?;
            buf.write_all(&version.to_le_bytes())?;
        }

        // Trailing CRC32 over the entire payload (V3+)
        let crc = crc32fast::hash(&buf);
        buf.write_all(&crc.to_le_bytes())?;

        Ok(buf)
    }

    /// Deserialize a manifest from bytes.
    pub fn deserialize(data: &[u8]) -> io::Result<Self> {
        if data.len() < 24 {
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
        if version > MANIFEST_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported manifest version {}", version),
            ));
        }

        // V3+: verify trailing CRC32 before parsing the rest.
        if version >= 3 {
            if data.len() < 28 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "manifest too small for CRC",
                ));
            }
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

        // Segments
        let seg_count = read_u32(data, &mut pos)? as usize;
        let mut segments = Vec::with_capacity(seg_count);
        for _ in 0..seg_count {
            // V6: 56 bytes (adds schema_version), V5: 48, V1-V4: 40
            let min_seg_bytes = if version >= 6 {
                56
            } else if version >= 5 {
                48
            } else {
                40
            };
            if pos + min_seg_bytes > data.len() {
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
            let compaction_epoch = if version >= 5 {
                read_u64(data, &mut pos)?
            } else {
                0
            };
            let schema_version = if version >= 6 {
                read_u64(data, &mut pos)?
            } else {
                0
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
                compaction_epoch,
                schema_version,
            });
        }

        // Tombstones (V2+, optional for V1 backward compat)
        // V4: (row_id, commit_seq) pairs. V2-V3: row_id only (commit_seq=0).
        // For V3+, the last 4 bytes are CRC, so stop before them.
        let tombstone_data_end = if version >= 3 {
            data.len() - 4
        } else {
            data.len()
        };
        let mut tombstones = Vec::new();
        if version >= 2 && pos + 8 <= tombstone_data_end {
            let tombstone_count = read_u64(data, &mut pos)? as usize;
            tombstones.reserve(tombstone_count);
            if version >= 4 {
                // V4: each tombstone is (row_id: i64, commit_seq: u64) = 16 bytes
                for _ in 0..tombstone_count {
                    if pos + 16 > tombstone_data_end {
                        break;
                    }
                    let row_id = read_i64(data, &mut pos)?;
                    let commit_seq = read_u64(data, &mut pos)?;
                    tombstones.push((row_id, commit_seq));
                }
            } else {
                // V2-V3: each tombstone is just row_id, assign commit_seq=0
                // (always visible to all snapshots — pre-restart tombstones)
                for _ in 0..tombstone_count {
                    if pos + 8 > tombstone_data_end {
                        break;
                    }
                    tombstones.push((read_i64(data, &mut pos)?, 0));
                }
            }
        }

        // Column renames (V5+)
        let mut column_renames = Vec::new();
        if version >= 5 && pos + 4 <= tombstone_data_end {
            let rename_count = read_u32(data, &mut pos)? as usize;
            column_renames.reserve(rename_count);
            for _ in 0..rename_count {
                if pos + 2 > tombstone_data_end {
                    break;
                }
                let old_len = u16::from_le_bytes(data[pos..pos + 2].try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "truncated rename old_name len")
                })?) as usize;
                pos += 2;
                if pos + old_len > tombstone_data_end {
                    break;
                }
                let old_name = std::str::from_utf8(&data[pos..pos + old_len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                pos += old_len;
                if pos + 2 > tombstone_data_end {
                    break;
                }
                let new_len = u16::from_le_bytes(data[pos..pos + 2].try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "truncated rename new_name len")
                })?) as usize;
                pos += 2;
                if pos + new_len > tombstone_data_end {
                    break;
                }
                let new_name = std::str::from_utf8(&data[pos..pos + new_len])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                pos += new_len;
                column_renames.push((SmartString::from(old_name), SmartString::from(new_name)));
            }
        }

        // Dropped columns (V6+: name + schema_version; V5: name only → version=0)
        let mut dropped_columns = Vec::new();
        if version >= 5 && pos + 4 <= tombstone_data_end {
            let count = read_u32(data, &mut pos)? as usize;
            for _ in 0..count {
                if pos + 2 > tombstone_data_end {
                    break;
                }
                let nlen = u16::from_le_bytes(data[pos..pos + 2].try_into().map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "truncated dropped col name")
                })?) as usize;
                pos += 2;
                if pos + nlen > tombstone_data_end {
                    break;
                }
                let name = std::str::from_utf8(&data[pos..pos + nlen])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                pos += nlen;
                let drop_version = if version >= 6 {
                    read_u64(data, &mut pos)?
                } else {
                    // V5 didn't store drop version; use 0 so all old volumes are masked
                    // (conservative: masks everything, same as old behavior)
                    u64::MAX
                };
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
            let d = std::fs::File::open(parent)
                .map_err(|e| io::Error::other(format!("failed to open dir for fsync: {}", e)))?;
            d.sync_all()
                .map_err(|e| io::Error::other(format!("failed to fsync dir: {}", e)))?;
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
    let total: usize = segments.values().map(|cs| cs.volume.row_count).sum();
    if reusable_seen.capacity() < total {
        reusable_seen.reserve(total * 8 / 7 + 16 - reusable_seen.capacity());
    }
    // Process newest-first (seg_order is ascending, so iterate reversed)
    for &seg_id in seg_order.iter().rev() {
        if let Some(cs) = segments.get_mut(&seg_id) {
            let rc = cs.volume.row_count;
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
                if !reusable_seen.insert(cs.volume.row_ids[i]) {
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
    // Clear but keep allocation for reuse on next seal/compact.
    reusable_seen.clear();
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
        let tombstone_map: FxHashMap<i64, u64> = manifest.tombstones.iter().copied().collect();
        Self {
            table_name,
            manifest: RwLock::new(manifest),
            segments: RwLock::new(Arc::new(FxHashMap::default())),
            volume_dir,
            has_segments_flag: std::sync::atomic::AtomicBool::new(false),
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

    /// Get all loaded segments (for scanning).
    pub fn get_segments(&self) -> Vec<Arc<FrozenVolume>> {
        let segs = Arc::clone(&*self.segments.read());
        segs.values().map(|cs| Arc::clone(&cs.volume)).collect()
    }

    /// Get segments in order (by segment_id, oldest first).
    pub fn get_segments_ordered(&self) -> Vec<Arc<FrozenVolume>> {
        // Hold manifest lock while cloning segments Arc to prevent
        // compaction from swapping the map between the two reads.
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
        // Hold manifest lock while cloning segments Arc to prevent
        // compaction's replace_segments_atomic from swapping the map
        // between the two reads (TOCTOU → missing rows in scan).
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

    /// Check if there are any segments. O(1) atomic read, no lock.
    pub fn has_segments(&self) -> bool {
        self.has_segments_flag
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Capture an atomic snapshot of segment state for batch constraint checking.
    /// Acquires manifest + segments + tombstones locks once. All per-row checks
    /// within the batch reuse this snapshot with zero lock overhead.
    pub fn cold_snapshot(&self) -> ColdSnapshot {
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
            if col_idx >= vol.columns.len() || col_idx >= vol.zone_maps.len() {
                continue;
            }
            if !vol.zone_maps[col_idx].may_contain_eq(value) {
                continue;
            }
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
                if vol.is_sorted(col_idx) {
                    // Always use OnceLock path: decompresses the full column
                    // once (~50ms) then caches it. compressed_store.binary_search_ge
                    // decompresses a fresh group per call (~0.3ms each), which is
                    // 100x slower for bulk validation (25K calls = 8+ seconds).
                    let start = vol.columns[col_idx].binary_search_ge(target);
                    let mut i = start;
                    while i < vol.row_count && vol.columns[col_idx].get_i64(i) == target {
                        let rid = vol.row_ids[i];
                        if seen.insert(rid) && !ts.contains_key(&rid) {
                            // Verify this is the authoritative version when overlapping
                            // volumes exist. After UPDATE changes PK + seal, the old PK
                            // value exists in an older volume without a tombstone.
                            if seg_ids.len() > 1 {
                                if let Some(current_row) = self.get_cold_row(rid) {
                                    if let Some(current_val) = current_row.get(col_idx) {
                                        if current_val != value {
                                            i += 1;
                                            continue; // stale PK in older volume
                                        }
                                    }
                                }
                            }
                            return Some(rid);
                        }
                        i += 1;
                    }
                } else {
                    for i in 0..vol.row_count {
                        let rid = vol.row_ids[i];
                        if !seen.insert(rid) {
                            continue;
                        }
                        if !vol.columns[col_idx].is_null(i)
                            && vol.columns[col_idx].get_i64(i) == target
                            && !ts.contains_key(&rid)
                        {
                            if seg_ids.len() > 1 {
                                if let Some(current_row) = self.get_cold_row(rid) {
                                    if let Some(current_val) = current_row.get(col_idx) {
                                        if current_val != value {
                                            continue; // stale PK in older volume
                                        }
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
                if vi < vol.zone_maps.len() && !vol.zone_maps[vi].may_contain_eq(values[i]) {
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
                if vi < vol.bloom_filters.len()
                    && !vol.bloom_filters[vi].might_contain_hash(bloom_hashes[i])
                {
                    bloom_skip = true;
                    break;
                }
            }
            if bloom_skip {
                continue;
            }

            // Tier 3: Per-volume hash index
            let mut vol_result: Option<i64> = None;
            if !has_missing {
                // Common path: no schema evolution, pass values directly (zero alloc)
                vol.unique_lookup_all(&vol_col_indices, values, |row_idx| {
                    let rid = vol.row_ids[row_idx as usize];
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
                for i in 0..vol.row_count {
                    let rid = vol.row_ids[i];
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
                    if let Some(current_row) = self.get_cold_row(rid) {
                        let still_matches = col_indices.iter().enumerate().all(|(i, &ci)| {
                            let v = current_row
                                .get(ci)
                                .cloned()
                                .unwrap_or_else(|| column_defaults[i].clone());
                            !v.is_null() && v == *values[i]
                        });
                        if !still_matches {
                            continue; // stale value in older volume, skip
                        }
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

    /// Get the oldest compaction_epoch across all segments.
    pub fn oldest_segment_epoch(&self) -> Option<u64> {
        let manifest = self.manifest.read();
        manifest.segments.iter().map(|s| s.compaction_epoch).min()
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
    pub fn commit_pending_tombstones(&self, txn_id: i64, commit_seq: u64) {
        let pending = self.pending_txn_tombstones.write().remove(&txn_id);
        if let Some(ids) = pending {
            if !ids.is_empty() {
                let id_vec: Vec<i64> = ids.into_iter().collect();
                self.add_tombstones(&id_vec, commit_seq);
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
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return false;
        }
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
                if cold.volume.row_ids.binary_search(&row_id).is_ok() {
                    return true;
                }
            }
        }
        false
    }

    /// Get a cold row by row_id. Returns the Row if found and not tombstoned.
    /// Iterates newest-first so overlapping row_ids return the newest version.
    pub fn get_cold_row(&self, row_id: i64) -> Option<crate::core::Row> {
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return None;
        }
        // Hold manifest lock while cloning segments Arc to prevent
        // compaction from swapping the map between the two reads.
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
                if let Ok(idx) = cold.volume.row_ids.binary_search(&row_id) {
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
    pub fn get_cold_row_normalized(
        &self,
        row_id: i64,
        schema: &crate::core::Schema,
    ) -> Option<crate::core::Row> {
        let ts = Arc::clone(&*self.tombstones.read());
        if ts.contains_key(&row_id) {
            return None;
        }
        // Hold manifest lock while cloning segments Arc to prevent
        // compaction from swapping the map between the two reads.
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
                if let Ok(idx) = cold.volume.row_ids.binary_search(&row_id) {
                    let mapping = self.get_volume_mapping(*seg_id, schema);
                    if mapping.is_identity {
                        return Some(cold.volume.get_row(idx));
                    }
                    return Some(cold.volume.get_row_mapped(idx, &mapping));
                }
            }
        }
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
                if cold.volume.row_ids.binary_search(&row_id).is_ok() {
                    return true;
                }
            }
        }
        false
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
        if segments.len() == 1 {
            let total: usize = segments.values().map(|cs| cs.volume.row_count).sum();
            return total.saturating_sub(tombstones.len());
        }

        // Fast path: use visibility bitmaps (O(1) per row, zero allocation).
        // visible=None means all rows visible (no overlap), is_visible() handles both.
        {
            let mut count = 0usize;
            for cs in segments.values() {
                let vol = &cs.volume;
                for i in 0..vol.row_count {
                    if !cs.is_visible(i) {
                        continue;
                    }
                    if !tombstones.is_empty() && tombstones.contains_key(&vol.row_ids[i]) {
                        continue;
                    }
                    count += 1;
                }
            }
            count
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
    pub fn segments_snapshot(&self) -> Arc<FxHashMap<u64, ColdSegment>> {
        Arc::clone(&*self.segments.read())
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
            compaction_epoch: 0,
            schema_version: 0,
        });
        m.add_segment(SegmentMeta {
            segment_id: 2,
            file_path: PathBuf::from("vol_002.vol"),
            row_count: 500,
            min_row_id: 1001,
            max_row_id: 1500,
            creation_lsn: 200,
            compaction_epoch: 0,
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
            compaction_epoch: 0,
            schema_version: 0,
        });
        m.add_segment(SegmentMeta {
            segment_id: 2,
            file_path: PathBuf::from("b.vol"),
            row_count: 100,
            min_row_id: 101,
            max_row_id: 200,
            creation_lsn: 0,
            compaction_epoch: 0,
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
            compaction_epoch: 0,
            schema_version: 3,
        });
        m.add_segment(SegmentMeta {
            segment_id: 3,
            file_path: PathBuf::from("seg_0003.vol"),
            row_count: 5000,
            min_row_id: 10001,
            max_row_id: 15000,
            creation_lsn: 30,
            compaction_epoch: 0,
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
            compaction_epoch: 0,
            schema_version: 0,
        });

        m.write_to_disk(&path).unwrap();
        let loaded = TableManifest::read_from_disk(&path).unwrap();

        assert_eq!(loaded.table_name.as_str(), "disk_test");
        assert_eq!(loaded.segments.len(), 1);
        assert_eq!(loaded.tombstones, vec![(5, 0), (10, 0)]);
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
            compaction_epoch: 0,
            schema_version: 0,
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
                compaction_epoch: 0,
                schema_version: 0,
            },
            None,
        );

        // Tombstone row_id=5 (commit_seq=1)
        mgr.add_tombstones(&[5], 1);
        assert!(!mgr.row_exists(5));
        assert!(mgr.row_exists(4));
        assert!(mgr.row_exists(6));
        assert_eq!(mgr.total_row_count(), 9);
        assert!(mgr.is_tombstoned(5));
        assert!(!mgr.is_tombstoned(4));

        // Clear tombstones
        mgr.clear_tombstones();
        assert!(mgr.row_exists(5));
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
                    compaction_epoch: 0,
                    schema_version: 0,
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
            compaction_epoch: 0,
            schema_version: 0,
        });
        mgr.add_tombstones(&[5], 1);

        assert_eq!(mgr.segment_count(), 1);
        mgr.clear();
        assert_eq!(mgr.segment_count(), 0);
        assert!(mgr.tombstone_set_arc().is_empty());
    }
}
